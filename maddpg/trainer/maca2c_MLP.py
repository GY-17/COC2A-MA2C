import numpy as np
import tensorflow as tf


# ============================================================
# MACA2C Policy: Actor (multi-head) + MACA-style Q-critic
# ============================================================

class MACA2CPolicy(tf.keras.Model):
    """
    MACA2C: MA2C + MACA-style two-level credit assignment.
    - Actor: local obs -> LSTM -> multi-head logits
    - Critic: Q(s, x) = v_phi(s) + w_phi(s)^T x, linear in action input x
      where x can be:
        * joint one-hot action vector (for Q(s,a))
        * joint policy vector \bar{π} (for E[Q(s,a)] baselines)
    """

    def __init__(self,
                 obs_dim,
                 act_space_list,
                 global_obs_dim,
                 joint_action_dim,
                 n_lstm: int = 64,
                 name: str = "maca2c_policy"):
        super().__init__(name=name)

        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.joint_action_dim = joint_action_dim
        self.act_space_list = act_space_list
        self.num_heads = len(act_space_list)
        self.n_lstm = n_lstm
        self.actor_lstm_state = None   # 新增

        # ======== Actor: 局部观测 -> LSTM -> 多头 logits ========
        self.actor_fc1 = tf.keras.layers.Dense(128, activation='relu', name='actor_fc1')
        self.actor_fc2 = tf.keras.layers.Dense(128, activation='relu', name='actor_fc2')
        self.actor_lstm = tf.keras.layers.LSTM(
            n_lstm,
            return_sequences=True,
            return_state=True,
            name='actor_lstm'
        )

        self.actor_heads = []
        for i, space in enumerate(act_space_list):
            # 只要求有 .n 属性（与 MA2C 一致）
            head = tf.keras.layers.Dense(space.n, activation=None, name=f'actor_head_{i}')
            self.actor_heads.append(head)

        # ======== Critic: 状态编码 + 线性动作头 (MACA) ========
        self.critic_state_fc1 = tf.keras.layers.Dense(256, activation='relu', name='critic_state_fc1')
        self.critic_state_fc2 = tf.keras.layers.Dense(128, activation='relu', name='critic_state_fc2')

        # v_phi(s)：只依赖状态的标量
        self.critic_q_state_bias = tf.keras.layers.Dense(
            1, activation=None, name='critic_q_state_bias'
        )
        # w_phi(s)：依赖状态的动作权重向量，长度 = joint_action_dim
        self.critic_q_action_weight = tf.keras.layers.Dense(
            joint_action_dim, activation=None, name='critic_q_action_weight'
        )
    
    def reset_states(self):
        """在episode开始时重置LSTM状态"""
        self.actor_lstm_state = None

    # ----------------------------------------------------------------------
    # Actor: 前向（支持单步 / 序列）
    # ----------------------------------------------------------------------
    def call(self, obs, training: bool = False, initial_state=None):
        """
        obs:
            - shape = (B, obs_dim)      -> 自动视为 (B, 1, obs_dim)
            - shape = (B, T, obs_dim)   -> 多步序列
        return:
            logits_list: 长度 = num_heads，
                         每个元素形状 (B, T, A_head)
            lstm_state: (h, c)
        """
        x = tf.convert_to_tensor(obs, dtype=tf.float32)

        if len(x.shape) == 2:
            # (B, obs_dim) -> (B, 1, obs_dim)
            x = tf.expand_dims(x, axis=1)

        # 2 层 MLP 编码局部观测
        x = self.actor_fc1(x)
        x = self.actor_fc2(x)

        # 决定用哪个初始状态
        if initial_state is not None:
            lstm_out, h, c = self.actor_lstm(x, initial_state=initial_state, training=training)
        elif self.actor_lstm_state is not None:
            lstm_out, h, c = self.actor_lstm(x, initial_state=self.actor_lstm_state, training=training)
        else:
            lstm_out, h, c = self.actor_lstm(x, training=training)

        # 在线动作当前不在 GradientTape 内；停止梯度仅作为防御性处理，
        # 不改变隐藏状态的前向数值。
        if training:
            self.actor_lstm_state = [
                tf.stop_gradient(h),
                tf.stop_gradient(c),
            ]

        logits_list = []
        for head in self.actor_heads:
            logits = head(lstm_out)  # (B, T, A_head)
            logits_list.append(logits)

        return logits_list, (h, c)

    # ----------------------------------------------------------------------
    # Critic: MACA-style Q(s, x) = v(s) + w(s)^T x
    # ----------------------------------------------------------------------
    def compute_q(self, global_obs, action_vec):
        """
        global_obs: (gdim,) 或 (B, gdim)
        action_vec: (action_dim,) 或 (B, action_dim)
                    - one-hot 联合动作向量
                    - 联合策略概率向量 \bar{π}
                    - “混合”向量（本 agent 用 prob，其它 agent 用 one-hot）
        return:
            Q(s, x)，标量或 (B,) 向量
        """
        g = tf.convert_to_tensor(global_obs, dtype=tf.float32)
        x = tf.convert_to_tensor(action_vec, dtype=tf.float32)

        if len(g.shape) == 1:
            g = tf.expand_dims(g, axis=0)  # (1, gdim)

        # 状态编码 z_s
        z = self.critic_state_fc1(g)
        z = self.critic_state_fc2(z)

        v_s = self.critic_q_state_bias(z)         # (B, 1)
        w_s = self.critic_q_action_weight(z)      # (B, joint_action_dim)

        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)         # (1, joint_action_dim)

        q = v_s + tf.reduce_sum(w_s * x, axis=-1, keepdims=True)  # (B, 1)
        return tf.squeeze(q, axis=-1)             # 标量或 (B,)

    # ----------------------------------------------------------------------
    # 便捷函数：构造联合动作 / 联合策略向量
    # ----------------------------------------------------------------------
    @staticmethod
    def build_joint_action_vector(actions_onehot_list):
        """拼接多个智能体的动作 one-hot，得到联合动作向量"""
        tensors = [tf.convert_to_tensor(v, dtype=tf.float32) for v in actions_onehot_list]
        return tf.concat(tensors, axis=-1)

    @staticmethod
    def build_joint_policy_vector(agent_prob_list):
        """拼接多个智能体的动作概率向量，得到联合策略向量 \bar{π}"""
        tensors = [tf.convert_to_tensor(v, dtype=tf.float32) for v in agent_prob_list]
        return tf.concat(tensors, axis=-1)

    # ----------------------------------------------------------------------
    # MACA: 联合 / 个体基线，全都复用同一个 Q 网络
    # ----------------------------------------------------------------------
    def compute_joint_baseline(self, global_obs, pi_joint):
        """
        联合基线：
            b^{Jnt}(s) = Q(s, \bar{π}^{Jnt})
        """
        return self.compute_q(global_obs, pi_joint)

    def compute_individual_baseline(self, global_obs, pi_mixed):
        """
        个体基线：
            b^{Ind}_i(s,a) = Q(s, \bar{π}_{G={i}})
        其中 \bar{π}_{G={i}} 的构造是：
            - 本 agent i 的那一段用概率向量 π_i
            - 其它 agent j≠i 的那一段用 one-hot(a_j)
        """
        return self.compute_q(global_obs, pi_mixed)


# ============================================================
# On-policy Buffer（MACA2C 专用）
# ============================================================

class OnPolicyBuffer:
    """
    存储单个 agent 的 on-policy 轨迹
    注意：joint action / joint policy 由外部多智能体管理逻辑在 update() 调用时提供
    """
    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma
        self.reset()

    def __len__(self):
        return len(self.rewards)

    def reset(self):
        self.obs = []          # 局部观测 o_t
        self.global_obs = []   # 全局观测 s_t
        self.acts = []         # 每步该 agent 的多头离散动作索引列表 [a_head0, a_head1, ...]
        self.rewards = []      # 标量 reward_t
        self.dones = []        # done 标记
        self.masks = []        # IAM 非法动作掩码（只对应 head0），可以为 None

    def add(self, obs, global_obs, acts, reward, done, mask=None):
        """
        obs:        (obs_dim,)
        global_obs: (global_obs_dim,)
        acts:       list/array[int]，长度 = num_heads
        mask:       IAM 掩码（针对 head0 的动作维度），可以为 None
        """
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.global_obs.append(np.asarray(global_obs, dtype=np.float32))
        self.acts.append(np.asarray(acts, dtype=np.int32))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.masks.append(mask)

    def get_returns(self, final_value: float = 0.0):
        """
        计算 MC 返回：
            R_t = r_t + γ R_{t+1},   如果 done_t=True，则 R_t = r_t
        """
        returns = []
        R = float(final_value)
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + self.gamma * R * (1.0 - float(done))
            returns.insert(0, R)
        return np.asarray(returns, dtype=np.float32)


# ============================================================
# MACA2C Agent Trainer（每个智能体一套 Actor + Critic）
# ============================================================

class MACA2CAgentTrainer:
    """
    多智能体 A2C + MACA 两级显示信用分配
    - 每个智能体各自拥有一套 MACA2CPolicy（Actor + Q-critic）
    - Critic 接受联合动作向量 / 联合策略向量，计算：
        Q(s,a), b^{Jnt}(s), b^{Ind}_i(s,a)
    - Actor 使用 IAM 非法动作掩码（仅作用于 head0，即“节点选择”那一头）
    """

    def __init__(self,
                 agent_index: int,
                 obs_dim: int,
                 global_obs_dim: int,
                 act_space_list,
                 joint_action_dim: int,
                 args):
        """
        agent_index: 当前智能体在多智能体系统中的索引（0 ~ N-1）
        obs_dim:     当前智能体的局部观测维度
        global_obs_dim: 全局观测维度
        act_space_list: 当前智能体的多头离散动作空间列表（每个有 .n）
        joint_action_dim: 所有智能体所有 head 的动作 one-hot 拼完之后的总维度
        args: 配置对象，需要属性：
            - gamma
            - lr_actor
            - lr_critic
            - psi_joint
            - psi_ind
            - use_iam
            - max_grad_norm（可选）
        """
        self.agent_index = agent_index
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.act_space_list = act_space_list
        self.num_heads = len(act_space_list)
        self.joint_action_dim = joint_action_dim

        self.args = args
        self.gamma = float(getattr(args, 'gamma', 0.95))
        self.psi_joint = float(getattr(args, 'psi_joint', 0.5))
        self.psi_ind = float(getattr(args, 'psi_ind', 0.5))
        self.use_iam = bool(getattr(args, 'use_iam', False))
        self.max_grad_norm = getattr(args, 'max_grad_norm', None)
        # 熵系数：会在 train.py 中按步数通过 setattr(tr, "entropy_coef", ...) 动态覆盖，实现熵退火
        self.entropy_coef = float(getattr(args, 'entropy_coef', 1e-3))


        # ===== Policy 网络 =====
        self.policy = MACA2CPolicy(
            obs_dim=obs_dim,
            act_space_list=act_space_list,
            global_obs_dim=global_obs_dim,
            joint_action_dim=joint_action_dim,
            n_lstm=getattr(args, 'num_units', 64),
            name=f'maca2c_policy_agent{agent_index}'
        )

        # 通过一次 dummy 前向，确保变量创建
        dummy_obs = tf.zeros((1, 1, obs_dim), dtype=tf.float32)
        _logits, _ = self.policy(dummy_obs, training=False)
        dummy_global = tf.zeros((1, global_obs_dim), dtype=tf.float32)
        dummy_action_vec = tf.zeros((1, joint_action_dim), dtype=tf.float32)
        _ = self.policy.compute_q(dummy_global, dummy_action_vec)

        # ===== 优化器 & 变量分组 =====
        lr_actor = float(getattr(args, 'lr_actor', getattr(args, 'lr', 1e-4)))
        lr_critic = float(getattr(args, 'lr_critic', getattr(args, 'lr', 3e-4)))

        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=lr_critic)

        # 变量分组：按名字里的前缀粗略分
        self.actor_vars = [v for v in self.policy.trainable_variables if 'actor_' in v.name]
        self.critic_vars = [v for v in self.policy.trainable_variables if 'critic_' in v.name]

        if not self.actor_vars or not self.critic_vars:
            print(f"[MACA2C][Warn] actor_vars={len(self.actor_vars)}, critic_vars={len(self.critic_vars)}")

        # on-policy 经验缓冲区
        self.buffer = OnPolicyBuffer(gamma=self.gamma)

    # ------------------------------------------------------------------
    # 采样动作（带 IAM 掩码），返回：动作索引列表、多头 one-hot、概率向量
    # ------------------------------------------------------------------
    def action(self, obs, mask=None):
        """
        obs:  当前智能体局部观测，shape=(obs_dim,)
        mask: IAM 非法动作掩码（仅应用于 head0，对应“节点选择”那一头），
              若为 None 或 args.use_iam=False，则不应用掩码。
              期望形状 = (num_nodes,) 或可转换为该长度的布尔向量。
        return:
            - actions_per_head: list[int]，每个 head 的离散动作索引
            - onehot_agent:     np.ndarray, shape=(sum_i A_i,)
            - pi_agent:         np.ndarray, 同上一维度，对应各 head 拼接后的概率向量
        """
        obs_tensor = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)  # (1, obs_dim)
        logits_list, _ = self.policy(obs_tensor, training=True)          # 每个 logits: (1, 1, A_i)

        # 统一 logits 形状为 (1, A_i)
        processed_logits = []
        for logits in logits_list:
            if len(logits.shape) == 3:
                # (B, T, A) -> (B, A)，此处 B=1, T=1
                logits = tf.squeeze(logits, axis=1)
            elif len(logits.shape) == 2:
                # 已是 (B, A)
                pass
            else:
                logits = tf.reshape(logits, [1, -1])
            processed_logits.append(logits)

        # ===== 应用 IAM 掩码（仅 head0）=====
        masked_logits_list = []
        if self.use_iam and mask is not None and len(processed_logits) > 0:
            for head_idx, logits in enumerate(processed_logits):
                if head_idx == 0:
                    mask_arr = tf.convert_to_tensor(mask, dtype=tf.bool)
                    # 若全 False，则降级为全 True，避免完全无梯度
                    if not tf.reduce_any(mask_arr):
                        mask_arr = tf.ones_like(mask_arr, dtype=tf.bool)
                    # 扩展 batch 维 -> (1, A)
                    mask_tensor = tf.expand_dims(mask_arr, axis=0)
                    masked_logits = tf.where(
                        mask_tensor,
                        logits,
                        tf.fill(tf.shape(logits), tf.constant(-1e9, tf.float32))
                    )
                    masked_logits_list.append(masked_logits)
                else:
                    masked_logits_list.append(logits)
        else:
            masked_logits_list = processed_logits

        # ===== 按 head 采样动作 =====
        actions = []
        probs_concat = []
        onehots_concat = []

        for logits in masked_logits_list:
            # logits: (1, A)
            logits_1d = tf.squeeze(logits, axis=0)          # (A,)
            probs = tf.nn.softmax(logits_1d).numpy()        # (A,)
            # 数值安全：若因溢出导致全 NaN，则降级为均匀
            if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
                probs = np.ones_like(probs, dtype=np.float32) / float(len(probs))
            else:
                probs = probs / probs.sum()

            a = int(np.random.choice(len(probs), p=probs))
            actions.append(a)

            onehot = np.zeros_like(probs, dtype=np.float32)
            onehot[a] = 1.0
            onehots_concat.append(onehot)
            probs_concat.append(probs)

        onehot_agent = np.concatenate(onehots_concat, axis=-1).astype(np.float32)
        pi_agent = np.concatenate(probs_concat, axis=-1).astype(np.float32)

        return actions, onehot_agent, pi_agent

    # ------------------------------------------------------------------
    # 存储经验
    # ------------------------------------------------------------------
    def experience(self, obs, global_obs, actions_per_head, reward, done, mask=None):
        """
        存一条 (o_t, s_t, a_t, r_t, done_t, mask_t) 进本 agent 的 on-policy buffer
        actions_per_head: list[int], 与 action() 返回的第一项对应
        """
        self.buffer.add(obs, global_obs, actions_per_head, reward, done, mask)
        if done:
            # 一个episode结束，重置LSTM记忆
            self.policy.reset_states()

    # ------------------------------------------------------------------
    # 更新 Actor / Critic（单个 agent），需由外部提供多智能体 joint 信息
    # ------------------------------------------------------------------
    def update(self,
            joint_action_onehots,
            joint_policies,
            all_agents_onehots,
            all_agents_probs):
        """
        joint_action_onehots: list[ np.ndarray ]，长度 = T
            - 第 t 项是该时间步所有智能体 one-hot 拼接得到的联合动作向量，shape=(joint_action_dim,)
        joint_policies: list[ np.ndarray ]，长度 = T
            - 第 t 项是该时间步所有智能体 multi-head 概率拼接得到的联合策略向量 \bar{π}^{Jnt}_t
        all_agents_onehots: list[ list[np.ndarray] ]，外层长度=T，内层长度= num_agents
            - all_agents_onehots[t][j] 是第 t 步第 j 个智能体的 one-hot 动作向量
        all_agents_probs:    list[ list[np.ndarray] ]，结构同上
            - all_agents_probs[t][j] 是第 t 步第 j 个智能体的 multi-head 概率向量

        返回：
            actor_loss, critic_loss（python float）
        """
        if len(self.buffer) == 0:
            return None, None

        T = len(self.buffer)
        assert T == len(joint_action_onehots) == len(joint_policies), \
            "MACA2C.update: 时间步数量不一致"

        # ===== 准备数据 =====
        obs_batch = np.stack(self.buffer.obs, axis=0)              # (T, obs_dim)
        global_obs_batch = np.stack(self.buffer.global_obs, axis=0)  # (T, gdim)
        acts_batch = np.stack(self.buffer.acts, axis=0)            # (T, num_heads)
        masks_batch = self.buffer.masks

        # ===== 1) 计算 MC returns =====
        returns = self.buffer.get_returns(final_value=0.0)         # (T,)

        # ===== 2) 计算两级 MACA 优势 A_i(t) =====
        advantages = []

        for t in range(T):
            s_t = global_obs_batch[t]
            a_joint = joint_action_onehots[t]
            pi_joint = joint_policies[t]

            # Q(s_t, a_t)
            q_sa = self.policy.compute_q(s_t, a_joint)

            # 联合基线 b^{Jnt}(s_t) = Q(s_t, \bar{π}^{Jnt}_t)
            b_jnt = self.policy.compute_joint_baseline(s_t, pi_joint)

            # 个体基线 b^{Ind}_i(s_t,a_t) = Q(s_t, \bar{π}_{G={i}}(t))
            # 构造 \bar{π}_{G={i}}：自己用 prob，其它 agent 用 one-hot
            mixed_parts = []
            for j, (onehot_j, prob_j) in enumerate(zip(all_agents_onehots[t], all_agents_probs[t])):
                if j == self.agent_index:
                    mixed_parts.append(tf.convert_to_tensor(prob_j, dtype=tf.float32))
                else:
                    mixed_parts.append(tf.convert_to_tensor(onehot_j, dtype=tf.float32))
            pi_mixed = tf.concat(mixed_parts, axis=-1)

            b_ind = self.policy.compute_individual_baseline(s_t, pi_mixed)

            adv = q_sa - (self.psi_joint * b_jnt + self.psi_ind * b_ind)
            advantages.append(float(adv.numpy() if isinstance(adv, tf.Tensor) else adv))

        advantages = np.asarray(advantages, dtype=np.float32)
        # 归一化优势，数值更稳定
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std < 1e-8:
            adv_std = 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ===== 3) Actor 更新（使用多头 logits + IAM 掩码 + A_i）=====
        obs_tensor = tf.convert_to_tensor(obs_batch[None, :, :], dtype=tf.float32)  # (1, T, obs_dim)
        acts_tensor = tf.convert_to_tensor(acts_batch, dtype=tf.int32)              # (T, num_heads)
        adv_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)            # (T,)

        # ---- 关键：保存并清空在线 LSTM 状态 ----
        saved_state = self.policy.actor_lstm_state
        self.policy.actor_lstm_state = None

        try:
            with tf.GradientTape() as actor_tape:
                actor_tape.watch(self.actor_vars)
                # 训练阶段用“干净”的 LSTM，不更新 actor_lstm_state
                logits_list, _ = self.policy(obs_tensor, training=False)            # 每个 logits: (1, T, A_head)

                total_logp = 0.0
                entropy_terms = []   # 新增：用于累计各 head 的熵

                for head_idx, logits in enumerate(logits_list):
                    # 统一 logits 形状为 (T, A)
                    if len(logits.shape) == 3:
                        # (B, T, A) -> (T, A)
                        logits = tf.squeeze(logits, axis=0)
                    elif len(logits.shape) == 2:
                        # (B, A) -> (1, A)，再视为 T=1
                        logits = tf.expand_dims(logits[0], axis=0)
                    else:
                        logits = tf.reshape(logits, [1, -1])

                    # ===== 训练阶段也应用 IAM 掩码（仅 head0）=====
                    if self.use_iam and head_idx == 0:
                        masked_logits_per_t = []
                        invalid_pick_count = 0
                        for t_step in range(T):
                            step_mask = masks_batch[t_step]
                            if step_mask is not None:
                                m = tf.convert_to_tensor(step_mask, dtype=tf.bool)
                                if not tf.reduce_any(m):
                                    m = tf.ones_like(m, dtype=tf.bool)
                                # 若该步实际选择的动作被 mask 禁用，记一下（方便 debug）
                                a_idx = acts_tensor[t_step, head_idx]
                                if int(a_idx.numpy()) >= 0 and (not bool(m[a_idx].numpy())):
                                    invalid_pick_count += 1
                                masked = tf.where(
                                    m,
                                    logits[t_step],
                                    tf.fill(tf.shape(logits[t_step]), tf.constant(-1e9, tf.float32))
                                )
                            else:
                                masked = logits[t_step]
                            masked_logits_per_t.append(masked)
                        logits = tf.stack(masked_logits_per_t, axis=0)  # (T, A)
                        if invalid_pick_count > 0:
                            print(f"[MACA2C][agent {self.agent_index}] head0 masked-pick count: "
                                  f"{invalid_pick_count}/{T}")

                    # log π(a|s) for this head
                    logp_all = tf.nn.log_softmax(logits, axis=-1)  # (T, A)
                    idx = tf.stack(
                        [tf.range(T, dtype=tf.int32), acts_tensor[:, head_idx]],
                        axis=1
                    )
                    logp_head = tf.gather_nd(logp_all, idx)        # (T,)
                    total_logp += logp_head

                    # ===== 熵项 H[π]，和 MA2C 写法保持一致 =====
                    probs = tf.nn.softmax(logits, axis=-1)
                    entropy = -tf.reduce_sum(probs * logp_all, axis=-1)   # (T,)
                    entropy_terms.append(tf.reduce_mean(entropy))

                # Policy gradient 部分（对优势视作常数权重）
                pg_loss = -tf.reduce_mean(total_logp * tf.stop_gradient(adv_tensor))

                # 所有 head 的平均熵
                if entropy_terms:
                    mean_entropy = tf.add_n(entropy_terms) / float(len(entropy_terms))
                else:
                    mean_entropy = tf.constant(0.0, dtype=tf.float32)

                # 最终用于反向传播的 Actor loss（带熵正则）
                actor_total_loss = pg_loss - self.entropy_coef * mean_entropy

            # 对 actor_total_loss 求梯度
            actor_grads = actor_tape.gradient(actor_total_loss, self.actor_vars)
            # 梯度裁剪（可选）
            if self.max_grad_norm is not None:
                new_grads = []
                for g in actor_grads:
                    if g is not None:
                        new_grads.append(tf.clip_by_norm(g, self.max_grad_norm))
                    else:
                        new_grads.append(None)
                actor_grads = new_grads
            self.actor_opt.apply_gradients(zip(actor_grads, self.actor_vars))

            # ===== 4) Critic 更新（Q(s,a) 回归 MC returns）=====
            with tf.GradientTape() as critic_tape:
                critic_tape.watch(self.critic_vars)
                q_preds = []
                for t in range(T):
                    s_t = global_obs_batch[t]
                    a_joint = joint_action_onehots[t]
                    q_val = self.policy.compute_q(s_t, a_joint)
                    q_preds.append(q_val)
                q_preds = tf.stack(q_preds)                       # (T,)
                ret_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
                critic_loss = tf.reduce_mean(tf.square(tf.stop_gradient(ret_tensor) - q_preds))

            critic_grads = critic_tape.gradient(critic_loss, self.critic_vars)
            if self.max_grad_norm is not None:
                new_grads = []
                for g in critic_grads:
                    if g is not None:
                        new_grads.append(tf.clip_by_norm(g, self.max_grad_norm))
                    else:
                        new_grads.append(None)
                critic_grads = new_grads
            self.critic_opt.apply_gradients(zip(critic_grads, self.critic_vars))

        finally:
            # ---- 恢复在线决策用的 LSTM 状态 ----
            self.policy.actor_lstm_state = saved_state

        # 清空 buffer，准备下一段轨迹
        self.buffer.reset()

        # 返回的还是“纯 PG loss”，方便和之前的曲线对比；
        # 真正反传用的是 actor_total_loss（含熵正则）
        return float(pg_loss.numpy()), float(critic_loss.numpy())
