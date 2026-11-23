# maddpg/trainer/comaddpg.py
import os
import numpy as np
import tensorflow as tf

from .model import ActorModel
from .per_buffer import PrioritizedReplayBuffer
from .utils import update_target_weights as soft_update

# 全局共享（中央 critic、target_critic、优化器、重放缓冲区）
_SHARED = {
    "built": False, "critic": None, "target_critic": None, "critic_opt": None, "replay": None,
    # ↓↓↓ 新增：用于把这一步每个 agent 的损失值共享出去
    "last_actor_losses": None,   # list[float]，长度 = n_agents
    "last_critic_loss": 0.0,     # float，这一轮 critic 的 loss
}


def _total_action_dim(n_agents, num_trucks, total_nodes, num_drones):
    # 和你环境一致：卡车(节点+是否等待)，无人机(服务节点+汇合卡车+是否继续)
    # 这里只需要平铺后的 total one-hot 维度即可
    dims = []
    for i in range(n_agents):
        if i < num_trucks:
            head = [total_nodes, 2]          # truck: target_node, wait
        else:
            head = [total_nodes, num_trucks, 2]  # drone: service_node, rendezvous_truck, continue
        dims.append(sum(head))
    return int(sum(dims))


def _head_dims_for_agent(agent_idx, num_trucks, total_nodes, num_trucks_for_rendez):
    if agent_idx < num_trucks:
        return [total_nodes, 2]
    else:
        return [total_nodes, num_trucks_for_rendez, 2]


def _concat_onehot_actions(ids_batch, head_dims_row):
    """
    ids_batch: int32 [B, H] -> concat one-hot [B, sum(head_dims_row)]
    """
    if ids_batch.shape.rank is not None and ids_batch.shape.rank >= 2:
        H_ids = ids_batch.shape[1]
        H_exp = len(head_dims_row)
        if (H_ids is not None) and (H_ids != H_exp):
            raise RuntimeError(f"[onehot] ids_batch has {H_ids} heads, "
                               f"but head_dims_row has {H_exp}: {head_dims_row}")
    B = tf.shape(ids_batch)[0]
    onehots = []
    for h, dim in enumerate(head_dims_row):
        idx = tf.cast(ids_batch[:, h], tf.int32)
        onehots.append(tf.one_hot(idx, depth=dim, dtype=tf.float32))
    return tf.concat(onehots, axis=1)  # [B, sum]


def _mask_logits(logits, mask_bool):
    if mask_bool is None:
        return logits
    huge_neg = tf.constant(-1e9, dtype=logits.dtype)
    mask_f = tf.cast(mask_bool, logits.dtype)     # True=1
    inv_f  = 1. - mask_f                          # False=1
    # 允许的位置保留 logits，禁止位置置为 huge_neg
    return logits * mask_f + huge_neg * inv_f


def _logprob_from_logits_and_ids(head_logits_list, actions_ids, iam_mask=None):
    # 多头 log-softmax 后，按各头选中的 id 聚合，再 across-head 求和
    logps = []
    for h, logits in enumerate(head_logits_list):
        if h == 0 and iam_mask is not None:
            logits = _mask_logits(logits, iam_mask)
        logp = tf.nn.log_softmax(logits, axis=1)
        a_h = tf.cast(actions_ids[:, h], tf.int32)
        gathered = tf.gather(logp, indices=a_h, batch_dims=1)  # [B]
        logps.append(gathered)
    return tf.add_n(logps)  # [B]


def _entropy_from_logits(head_logits_list, iam_mask=None):
    ents = []
    for h, logits in enumerate(head_logits_list):
        if h == 0 and iam_mask is not None:
            logits = _mask_logits(logits, iam_mask)
        p = tf.nn.softmax(logits, axis=1)
        ent = -tf.reduce_sum(p * tf.math.log(p + 1e-8), axis=1)  # [B]
        ents.append(ent)
    return tf.add_n(ents) / float(len(head_logits_list))  # [B]


def _sample_actions_from_logits(head_logits_list, iam_mask=None):
    # 从各 head 的 softmax 采样一个 action id，返回 [B, H]
    ids = []
    for h, logits in enumerate(head_logits_list):
        if h == 0 and iam_mask is not None:
            logits = _mask_logits(logits, iam_mask)
        # 直接对 logits 做 categorical 采样更稳定
        sampled = tf.random.categorical(logits, num_samples=1)  # [B,1]
        ids.append(tf.squeeze(sampled, axis=1))
    return tf.stack(ids, axis=1)  # [B,H]


def _ensure_shared(n_agents, global_obs_dim, total_act_dim, args):
    if _SHARED["built"]:
        return
    input_dim = int(global_obs_dim + total_act_dim)

    # 多头中央 critic，一次前向出 n_agents 个 Q_i
    class MultiHeadCentralCritic(tf.keras.Model):
        def __init__(self, input_dim, hidden_units=(1024, 512, 256),
                     activation='relu', n_heads=1):
            super().__init__()
            self.layers_ = []
            for u in hidden_units:
                self.layers_.append(tf.keras.layers.Dense(u, activation=activation))
            self.out = tf.keras.layers.Dense(n_heads)

        def call(self, x):
            y = x
            for lyr in self.layers_:
                y = lyr(y)
            return self.out(y)  # [B, n_agents]

    _SHARED["critic"] = MultiHeadCentralCritic(
        input_dim=input_dim,
        hidden_units=getattr(args, "hidden_units_critic", (1024, 512, 256)),
        activation=getattr(args, "activation", "relu"),
        n_heads=n_agents
    )
    _SHARED["critic"].build((None, input_dim))

    _SHARED["target_critic"] = MultiHeadCentralCritic(
        input_dim=input_dim,
        hidden_units=getattr(args, "hidden_units_critic", (1024, 512, 256)),
        activation=getattr(args, "activation", "relu"),
        n_heads=n_agents
    )
    _SHARED["target_critic"].build((None, input_dim))
    _SHARED["target_critic"].set_weights(_SHARED["critic"].get_weights())

    _SHARED["critic_opt"] = tf.keras.optimizers.Adam(getattr(args, "lr_critic", 1e-3))
    _SHARED["critic_input_dim"] = input_dim   # ← 新增：记住期望的输入维度

    # 共享 PER（或退化为均匀采样）
    if _SHARED["replay"] is None:
        cap = int(getattr(args, "buffer_size", 50000))
        use_per = bool(getattr(args, "use_per", True))
        if use_per:
            _SHARED["replay"] = PrioritizedReplayBuffer(
                capacity=cap,
                mu=getattr(args, "per_mu", 0.6),
                sigma=getattr(args, "per_sigma", 1.0),
                eta=getattr(args, "per_eta", 1e-6),
                beta=getattr(args, "per_beta", 0.4),
            )
        else:
            _SHARED["replay"] = PrioritizedReplayBuffer(capacity=cap)

    _SHARED["built"] = True  # ✅ 别再覆盖！

# ======================================================================
#  Trainer
# ======================================================================

class COMADDPGAgentTrainer:
    """
    去中心化 actor + 共享多头中央 critic + COMA 采样式反事实优势
    """

    def __init__(self, name, obs_shape_n, act_space_list, agent_index, args,
                 global_obs_dim=None, num_trucks=None, num_drones=None, total_nodes=None):
        self.name = name
        self.agent_index = agent_index
        self.n = len(obs_shape_n)
        self.args = args

        # 环境拓扑
        self.num_trucks = num_trucks if num_trucks is not None else getattr(args, 'num_trucks', 3)
        self.num_drones = num_drones if num_drones is not None else getattr(args, 'num_drones', 5)
        self.total_nodes = total_nodes if total_nodes is not None else getattr(args, 'total_nodes', 6)

        # 本体观测维度
        def _obs_dim(i):
            s = obs_shape_n[i]
            return s[0] if isinstance(s, tuple) else int(s)
        self.obs_dim = _obs_dim(agent_index)

        # 本体多头布置
        self.head_dims = _head_dims_for_agent(agent_index, self.num_trucks, self.total_nodes, self.num_trucks)
        self.act_dim_flat = int(sum(self.head_dims))

        # Actor / Target-Actor（多头离散，每头输出 logits）
        self.actor = ActorModel(self.obs_dim,
                                [type('Discrete', (), {'n': d})() for d in self.head_dims],
                                hidden_units=getattr(args, 'hidden_units_actor', (512, 256)),
                                activation=getattr(args, 'activation', 'relu'))
        self.actor.build((None, self.obs_dim))

        self.target_actor = ActorModel(self.obs_dim,
                                       [type('Discrete', (), {'n': d})() for d in self.head_dims],
                                       hidden_units=getattr(args, 'hidden_units_actor', (512, 256)),
                                       activation=getattr(args, 'activation', 'relu'))
        self.target_actor.build((None, self.obs_dim))
        self.target_actor.set_weights(self.actor.get_weights())

        self.actor_opt = tf.keras.optimizers.Adam(getattr(args, 'lr_actor', 1e-4))

        # IAM / 掩码、探索、熵系数
        self.use_iam = getattr(args, "use_iam", True)
        self.epsilon = getattr(args, "epsilon", 0.05)
        self.entropy_coef = float(getattr(args, "entropy_coef", 1e-3))  # 允许 train.py 动态覆盖

        # 共享 critic & PER
        tot_act_dim = _total_action_dim(self.n, self.num_trucks, self.total_nodes, self.num_drones)
        _ensure_shared(n_agents=self.n,
                       global_obs_dim=int(global_obs_dim),
                       total_act_dim=int(tot_act_dim),
                       args=args)

        self.gamma = getattr(args, "gamma", 0.99)
        self.tau   = getattr(args, "tau",   0.01)

        # 反事实采样配置
        self.cf_K = int(getattr(args, "cf_K", 8))
        self.cf_use_target = bool(getattr(args, "cf_use_target", True))
        self.baseline_masked = bool(getattr(args, "baseline_masked", True))

    # ---------- acting ----------
    def action(self, obs, mask=None):
        obs_tf = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
        heads = self.actor(obs_tf)
        picked = []
        for h, logits in enumerate(heads):
            if h == 0 and self.use_iam and mask is not None:
                logits = _mask_logits(logits, tf.convert_to_tensor(mask[None, :], dtype=tf.bool))
            if np.random.rand() < self.epsilon:
                if h == 0 and self.use_iam and mask is not None:
                    allowed = np.where(mask)[0]
                    a = int(np.random.choice(allowed)) if len(allowed) > 0 else int(np.argmax(logits.numpy()[0]))
                else:
                    a = int(np.random.randint(0, logits.shape[1]))
            else:
                a = int(np.argmax(logits.numpy()[0]))
            picked.append(a)
        return picked

    # ---------- buffer I/O ----------
    def experience(self, obs_n, global_obs, act_n, rew, new_obs_n, new_global_obs,
               done, mask_n=None, new_mask_n=None):
        """
        期望一次性塞入“联合”转移（请在 train.py 每步只调用一次：用 trainers[0] 来 add）
        """

        def _stack_mask_2d(m):
            # 把 [n_agents] 的 list/tuple/array -> (n_agents, total_nodes) 的 bool array
            if m is None:
                return None
            if isinstance(m, (list, tuple)):
                try:
                    arr = np.stack(
                        [np.asarray(x, dtype=bool) if x is not None else np.zeros((self.total_nodes,), dtype=bool)
                        for x in m],
                        axis=0
                    )  # (n_agents, total_nodes)
                    return arr
                except Exception:
                    return None
            m_arr = np.asarray(m)
            if m_arr.ndim == 1 and m_arr.shape[0] == self.total_nodes:
                # 单个向量 -> 复制 n_agents 份
                return np.repeat(m_arr[None, :], self.n, axis=0)
            if m_arr.ndim == 2:
                return m_arr.astype(bool)
            return None

        mask_n      = _stack_mask_2d(mask_n)
        new_mask_n  = _stack_mask_2d(new_mask_n)

        _SHARED["replay"].add(
            obs_n, global_obs, act_n, rew,
            new_obs_n, new_global_obs, done,
            mask_n, new_mask_n
        )

    def preupdate(self):
        pass

    # ---------- training ----------
    def update(self, trainers, t, target_update_freq=None):
        # ====== 非 0 号：不训练，但把“自己的 actor loss”从共享中取回用于日志 ======
        if self.agent_index != 0:
            al = _SHARED.get("last_actor_losses", None)
            if al is not None and len(al) == self.n:
                # critic 用 0.0（只显示一条共享 critic 曲线）
                return 0.0, float(al[self.agent_index])
            return 0.0, 0.0

        # 经验不足则跳过
        if self.buffer_size() < getattr(self.args, "batch_size", 256):
            return 0.0, 0.0

        # 从共享 PER 采样
        batch = _SHARED["replay"].sample(self.args.batch_size)
        # 正确顺序： ... dones, idxs, weights, masks, next_masks
        (obses, global_obses, actions, rewards,
        next_obses, next_global_obses, dones,
        idxs, isw, masks, next_masks) = batch

        # --- 规范化 batch 掩码到 [B, n_agents, total_nodes] 的 tf.bool ---
        def _normalize_mask_batch(m, B, n, total_nodes):
            if m is None:
                return None
            # 先试最直接的：能转 tensor 并且是 3D 就直接用
            try:
                t = tf.convert_to_tensor(m)
                if len(t.shape) == 3:
                    return tf.cast(t, tf.bool)
                # 如果是 2D (n_agents, total_nodes)，复制 B 份
                if len(t.shape) == 2 and int(t.shape[0]) == n:
                    t = tf.repeat(tf.cast(t[None, :, :], tf.bool), repeats=B, axis=0)
                    return t
            except Exception:
                pass
            # 如果是长度为 B 的 list/tuple，每个元素是 (n_agents, total_nodes)
            if isinstance(m, (list, tuple)) and len(m) == B:
                try:
                    arr = np.stack([np.asarray(x, dtype=bool).reshape(n, -1) for x in m], axis=0)
                    arr = arr[:, :n, :total_nodes]
                    return tf.convert_to_tensor(arr, dtype=tf.bool)
                except Exception:
                    return None
            # fallback：尝试转 ndarray 再兜底
            try:
                arr = np.asarray(m)
                if arr.ndim == 3:
                    arr = arr[:, :n, :total_nodes]
                    return tf.convert_to_tensor(arr, dtype=tf.bool)
                if arr.ndim == 2 and arr.shape[0] == n:
                    arr = np.repeat(arr[None, :, :total_nodes].astype(bool), B, axis=0)
                    return tf.convert_to_tensor(arr, dtype=tf.bool)
            except Exception:
                pass
            return None

        # B 取 batch 的真实大小（不要假设等于 args.batch_size）
        B = actions.shape[0] if hasattr(actions, "shape") else self.args.batch_size
        masks_tf      = _normalize_mask_batch(masks,      B, self.n, self.total_nodes)
        next_masks_tf = _normalize_mask_batch(next_masks, B, self.n, self.total_nodes)

        # ------------- Critic update -------------
        # 构造当前 joint one-hot 动作 & next 动作（由 target-actor 贪心/采样生成）
        joint_onehots = []
        next_joint_onehots = []

        for i in range(self.n):
            # ---- baseline b_i ≈ 1/K Σ Q_i(s, (ã_i, a_-i)) ----
            agent_obs_dim = int(getattr(trainers[i], "obs_dim", self.obs_dim))
            obs_i = tf.convert_to_tensor(obses[:, i, :agent_obs_dim], dtype=tf.float32)
            logits_heads = trainers[i].actor(obs_i)

            # ✅ 加这一行：i号智能体自己的 head 配置
            head_dims_i = _head_dims_for_agent(i, self.num_trucks, self.total_nodes, self.num_trucks)
            ids_i = tf.convert_to_tensor(actions[:, i, :len(head_dims_i)], dtype=tf.int32)  # ✅ 按头数裁切
            onehot_i = _concat_onehot_actions(ids_i, head_dims_i)
            joint_onehots.append(onehot_i)

            # agent 专属 obs 维度
            agent_obs_dim = int(getattr(trainers[i], "obs_dim", self.obs_dim))
            obs_i_next = tf.convert_to_tensor(next_obses[:, i, :agent_obs_dim], dtype=tf.float32)
            logits_heads = trainers[i].target_actor(obs_i_next)

            iam_mask = None
            if self.use_iam and next_masks_tf is not None:
                iam_mask = next_masks_tf[:, i, :]   # [B, total_nodes]，已是 tf.bool
            # 这里为了稳定 TD，使用贪心
            ids_next = []
            for h, logits in enumerate(logits_heads):
                if h == 0 and iam_mask is not None:
                    logits = _mask_logits(logits, iam_mask)
                ids_next.append(tf.argmax(logits, axis=1, output_type=tf.int32))
            ids_next = tf.stack(ids_next, axis=1)
            if ids_next.shape[1] != len(head_dims_i):
                raise RuntimeError(f"[next ids] agent{i}: got {ids_next.shape[1]} heads, expected {len(head_dims_i)}")
            next_onehot_i = _concat_onehot_actions(ids_next, head_dims_i)
            next_joint_onehots.append(next_onehot_i)

        joint_onehots = tf.concat(joint_onehots, axis=1)          # [B, total_act_dim]
        next_joint_onehots = tf.concat(next_joint_onehots, axis=1)

        critic_in_cur  = tf.concat([global_obses, joint_onehots], axis=1)
        critic_in_next = tf.concat([next_global_obses, next_joint_onehots], axis=1)

        # print("[COMA] critic_in_cur dim =", int(critic_in_cur.shape[1]),
        #         " expected =", int(_SHARED["critic_input_dim"]))
        Q_next_vec = _SHARED["target_critic"](critic_in_next)   # [B, n]

        # 联合奖励：这里用 sum(rew_i)（更符合协作目标）；dones 形状 [B,1] or [B] 都兼容
        r_joint = tf.reduce_sum(rewards, axis=1, keepdims=True) # [B,1]
        done_f  = tf.cast(dones, tf.float32)
        if len(done_f.shape) == 1:
            done_f = done_f[:, None]

        # gamma = float(self.gamma)
        # # y = r_joint + (1.0 - done_f) * gamma * tf.reduce_mean(Q_next_vec, axis=1, keepdims=True)  # 加权也可
        # # 建议：逐头目标，[B,n]
        # y_vec = r_joint + (1.0 - done_f) * gamma * Q_next_vec      # [B, n]
        # td_err_vec = y_vec - Q_vec_c                                # [B, n]
        # loss_c = mean(square(td_err_vec))  # 可带 IS 权重
        # y = tf.stop_gradient(y)  # TD 目标
        y_vec = r_joint + (1.0 - done_f) * self.gamma * Q_next_vec  # [B, n]
        y_vec = tf.stop_gradient(y_vec)

        with tf.GradientTape() as tape_c:
            Q_vec_c = _SHARED["critic"](critic_in_cur)               # [B, n]
            td_err_vec = y_vec - Q_vec_c                              # [B, n]
            if isw is not None:
                w = tf.reshape(tf.convert_to_tensor(isw, tf.float32), [-1, 1])  # [B,1]
                loss_c = tf.reduce_mean(w * tf.square(td_err_vec))
            else:
                loss_c = tf.reduce_mean(tf.square(td_err_vec))

        grads_c = tape_c.gradient(loss_c, _SHARED["critic"].trainable_variables)
        _SHARED["critic_opt"].apply_gradients(zip(grads_c, _SHARED["critic"].trainable_variables))

        # ★ PER：按这批样本的 TD 误差回写优先级（各头取均值或 L2 都可）
        try:
            pr = tf.reduce_mean(tf.abs(td_err_vec), axis=1).numpy()  # [B]
            _SHARED["replay"].update_priorities(idxs, pr)
        except Exception:
            pass

        # ------------- Actor update (per-agent) -------------
        # 给 actor 的优势用一个“停止梯度”的 Q_i(s,a)
        Q_i_all = tf.stop_gradient(_SHARED["critic"](critic_in_cur))  # [B,n]

        per_agent_actor_losses = []    # ← 新增：保存每个体的 loss_i
        total_actor_loss = 0.0
        for i in range(self.n):
            # === 该 agent 的真实 obs 维度 + head 规格 ===
            agent_obs_dim = int(getattr(trainers[i], "obs_dim", self.obs_dim))
            head_dims_i   = _head_dims_for_agent(i, self.num_trucks, self.total_nodes, self.num_trucks)

            # 当前步 obs_i（按各自 obs_dim 切片，避免 286 vs 292 错配）
            obs_i = tf.convert_to_tensor(obses[:, i, :agent_obs_dim], dtype=tf.float32)

            # IAM 掩码（只用于 head-0），若无则为 None
            iam_mask_i = masks_tf[:, i, :] if (self.use_iam and masks_tf is not None) else None

            # 执行过的动作 ids（来自回放），只取该 agent 的头数列
            ids_i_exec = tf.convert_to_tensor(actions[:, i, :len(head_dims_i)], dtype=tf.int32)

            # Q_i(s,a)（给优势用）
            q_i_sa = Q_i_all[:, i]  # [B]

            # === 采样式反事实基线：固定 a_-i，仅对 a_i 采样 K 次 ===
            actor_for_sample = trainers[i].target_actor if self.cf_use_target else trainers[i].actor
            q_cf_acc = 0.0
            K = int(self.cf_K)

            for _ in range(K):
                # 从 actor_for_sample 采样 a_i'（多头离散；对 head-0 应用 IAM）
                logits_i_samp = actor_for_sample(obs_i)  # list of [B, dim_h]
                ids_i_samp    = _sample_actions_from_logits(
                    logits_i_samp,
                    iam_mask=iam_mask_i if (self.use_iam and self.baseline_masked) else None
                )
                # 采样的“列数”必须等于 head 数
                if ids_i_samp.shape[1] != len(head_dims_i):
                    raise RuntimeError(f"[COMA] agent{i} sampled heads={ids_i_samp.shape[1]} "
                                    f"!= len(head_dims_i)={len(head_dims_i)}; head_dims_i={head_dims_i}")

                onehot_i_samp = _concat_onehot_actions(ids_i_samp, head_dims_i)   # [B, sum(head_dims_i)]

                # 其它 agent 的 one-hot（用各自 head 规格 + 执行过的动作）
                left, right = [], []
                for j in range(self.n):
                    if j == i:
                        continue
                    head_dims_j = _head_dims_for_agent(j, self.num_trucks, self.total_nodes, self.num_trucks)
                    ids_j_exec  = tf.convert_to_tensor(actions[:, j, :len(head_dims_j)], dtype=tf.int32)
                    onehot_j    = _concat_onehot_actions(ids_j_exec, head_dims_j)
                    (left if j < i else right).append(onehot_j)

                left  = tf.concat(left,  axis=1) if left  else tf.zeros([tf.shape(obs_i)[0], 0], tf.float32)
                right = tf.concat(right, axis=1) if right else tf.zeros([tf.shape(obs_i)[0], 0], tf.float32)

                joint_sampled = tf.concat([left, onehot_i_samp, right], axis=1)          # [B, total_act_dim]
                critic_in     = tf.concat([global_obses, joint_sampled], axis=1)         # [B, go+act]
                Qi_vec        = (_SHARED["target_critic"] if self.cf_use_target else _SHARED["critic"])(critic_in)  # [B,n]
                q_cf_acc     += Qi_vec[:, i]  # [B]

            b_i = q_cf_acc / float(K)               # [B]
            # A_i = tf.stop_gradient(q_i_sa - b_i)    # [B]
            # ---- 优势标准化（强烈建议保留；能显著抑制爆梯度）----
            A_i = q_i_sa - b_i                      # [B]
            # 标准化到 0-均值/1-方差，并适度截断
            A_i = (A_i - tf.reduce_mean(A_i)) / (tf.math.reduce_std(A_i) + 1e-6)
            A_i = tf.clip_by_value(A_i, -10.0, 10.0)
            A_i = tf.stop_gradient(A_i)

            # === log π_i(a_i|o_i)（多头 log-softmax 按真实动作索引聚合）+ 熵 ===
            with tf.GradientTape() as tape_a:
                heads_now = trainers[i].actor(obs_i)
                logpi_i   = _logprob_from_logits_and_ids(
                                heads_now, ids_i_exec,
                                iam_mask=iam_mask_i if (self.use_iam and self.baseline_masked) else None
                            )
                entropy_i = _entropy_from_logits(
                                heads_now,
                                iam_mask=iam_mask_i if (self.use_iam and self.baseline_masked) else None
                            )
                loss_i = -tf.reduce_mean(logpi_i * A_i) \
                        - float(self.entropy_coef) * tf.reduce_mean(entropy_i)
                        
            grads_i = tape_a.gradient(loss_i, trainers[i].actor.trainable_variables)
            grads_i = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads_i]
            trainers[i].actor_opt.apply_gradients(zip(grads_i, trainers[i].actor.trainable_variables))
            # total_actor_loss += float(loss_i.numpy())
            li = float(loss_i.numpy())
            per_agent_actor_losses.append(li)   # ← 逐个保存
            total_actor_loss += li


        # 软更新 target
        soft_update(_SHARED["critic"].variables, _SHARED["target_critic"].variables, self.tau)
        for tr in trainers:
            soft_update(tr.actor.variables, tr.target_actor.variables, self.tau)

        # ====== 把本轮“每个体 actor loss”和“共享 critic loss”写到共享 ======
        _SHARED["last_actor_losses"] = per_agent_actor_losses
        _SHARED["last_critic_loss"]  = float(loss_c.numpy())

        # 0 号也只返回“自己的” actor loss（而不是平均）
        return float(loss_c.numpy()), float(per_agent_actor_losses[self.agent_index])

    # ---------- misc ----------
    def buffer_size(self):
        try:
            return int(_SHARED["replay"].tree.n_entries)
        except Exception:
            return 0

    @property
    def buffer(self):
        class _Proxy:
            def __len__(self_inner):
                try:
                    return int(_SHARED["replay"].tree.n_entries)
                except Exception:
                    return 0
        return _Proxy()

    def save(self, save_dir, prefix=""):
        os.makedirs(save_dir, exist_ok=True)
        self.actor.save_weights(os.path.join(save_dir, f"actor_{self.agent_index}_{prefix}.h5"))
        self.target_actor.save_weights(os.path.join(save_dir, f"tactor_{self.agent_index}_{prefix}.h5"))

    def load(self, save_dir, prefix=""):
        self.actor.load_weights(os.path.join(save_dir, f"actor_{self.agent_index}_{prefix}.h5"))
        self.target_actor.load_weights(os.path.join(save_dir, f"tactor_{self.agent_index}_{prefix}.h5"))

    # 给 train.py 用来单次保存共享 critic
    def save_shared_critic(self, save_dir, prefix=""):
        os.makedirs(save_dir, exist_ok=True)
        _SHARED["critic"].save_weights(os.path.join(save_dir, f"shared_critic_{prefix}.h5"))
        _SHARED["target_critic"].save_weights(os.path.join(save_dir, f"shared_target_critic_{prefix}.h5"))
