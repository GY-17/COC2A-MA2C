import numpy as np
import tensorflow as tf


# ============================================================
# Critic 专用 Transformer Encoder Block
# ============================================================

class CriticTransformerBlock(tf.keras.layers.Layer):
    """
    一个标准的 Transformer Encoder Block：
    Multi-Head Self-Attention + FFN + 残差 + LayerNorm
    用于对 (B, L, d_model) 的 token 序列做编码。
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 name: str = "critic_block"):
        super().__init__(name=name)

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} 必须能被 num_heads={num_heads} 整除，"
                f"否则 MultiHeadAttention 的 key_dim 不为整数。"
            )

        # 多头自注意力
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            name=f"{name}_mha"
        )
        # 残差 + LayerNorm
        self.ln1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{name}_ln1"
        )

        # 前馈网络 (FFN)
        self.ffn_dense1 = tf.keras.layers.Dense(
            d_ff,
            activation='relu',
            name=f"{name}_ffn1"
        )
        self.ffn_dense2 = tf.keras.layers.Dense(
            d_model,
            activation=None,
            name=f"{name}_ffn2"
        )
        self.ln2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{name}_ln2"
        )

    def call(self, x, training: bool = False):
        """
        x: (B, L, d_model)
        """
        # Self-attention
        attn_out = self.mha(query=x, value=x, key=x, training=training)
        x = self.ln1(x + attn_out)

        # FFN
        ffn_out = self.ffn_dense2(self.ffn_dense1(x))
        x = self.ln2(x + ffn_out)

        return x

class MACA2CSharedCritic(tf.keras.Model):
    """
    多智能体共享的 MACA Q-critic：
        Q(s, x) = v(s) + w(s)^T x
    只负责：
        - 用 Transformer encoder 对 multi-token 状态编码
        - 输出 v(s), w(s)，供:
            * Q(s,a)（x = 联合 one-hot）
            * b^Jnt(s)（x = 联合策略概率向量）
            * b^Ind_i(s,a)（x = 混合向量）
    """
    def __init__(self,
                 global_obs_dim: int,
                 joint_action_dim: int,
                 critic_d_model: int = 128,
                 critic_num_heads: int = 4,
                 critic_num_layers: int = 2,
                 critic_d_ff: int = 256,
                 name: str = "maca2c_shared_critic"):
        super().__init__(name=name)
        self.global_obs_dim = global_obs_dim
        self.joint_action_dim = joint_action_dim

        self.critic_d_model = critic_d_model
        self.critic_num_heads = critic_num_heads
        self.critic_num_layers = critic_num_layers
        self.critic_d_ff = critic_d_ff

        # 把输入 token 映射到 d_model 维
        self.critic_input_proj = tf.keras.layers.Dense(
            self.critic_d_model,
            activation=None,
            name='shared_critic_input_proj'
        )

        # 多层 Transformer Encoder blocks
        self.critic_blocks = [
            CriticTransformerBlock(
                d_model=self.critic_d_model,
                num_heads=self.critic_num_heads,
                d_ff=self.critic_d_ff,
                name=f'shared_critic_block_{i}'
            )
            for i in range(self.critic_num_layers)
        ]

        # v(s), w(s)
        self.critic_q_state_bias = tf.keras.layers.Dense(
            1, activation=None, name='shared_critic_q_state_bias'
        )
        self.critic_q_action_weight = tf.keras.layers.Dense(
            joint_action_dim, activation=None, name='shared_critic_q_action_weight'
        )

        # # === Debug: 打印一次共享 critic 的结构参数 ===
        # print("\n[MACA2C][SharedCritic] Initialized.")
        # print(f"  global_obs_token_dim = {global_obs_dim}")
        # print(f"  joint_action_dim     = {joint_action_dim}")
        # print(f"  d_model={critic_d_model}, num_heads={critic_num_heads}, "
        #       f"num_layers={critic_num_layers}, d_ff={critic_d_ff}")
        
        # input("# === Debug: 打印一次共享 critic 的结构参数 ===")

    # ---- 状态编码：multi-agent Transformer encoder ----
    def _encode_state(self, global_obs, training: bool = False):
        """
        global_obs:
            - 通常是 (B, L, feat_dim)，你现在传的是 (1, n_agents, token_dim)
            - 也兼容 (L, feat_dim)，会自动视为 batch=1
        """
        g = tf.convert_to_tensor(global_obs, dtype=tf.float32)

        if len(g.shape) == 2:
            # (L, feat) -> (1, L, feat)
            g = tf.expand_dims(g, axis=0)
        elif len(g.shape) != 3:
            raise ValueError(f"global_obs 必须是 (L, feat) 或 (B, L, feat)，当前形状为 {g.shape}")

        x = self.critic_input_proj(g)          # (B, L, d_model)
        for block in self.critic_blocks:
            x = block(x, training=training)    # (B, L, d_model)

        z = tf.reduce_mean(x, axis=1)          # (B, d_model)
        return z

    def encode_state(self, global_obs, training: bool = False):
        """Expose the pooled critic state representation used by g_omega."""
        return self._encode_state(global_obs, training=training)

    # ---- 核心：Q(s,x) = v(s) + w(s)^T x ----
    def compute_q(self, global_obs, action_vec, training: bool = False):
        x = tf.convert_to_tensor(action_vec, dtype=tf.float32)
        z = self._encode_state(global_obs, training=training)

        v_s = self.critic_q_state_bias(z)         # (B,1)
        w_s = self.critic_q_action_weight(z)      # (B,D)

        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)         # (1,D)

        q = v_s + tf.reduce_sum(w_s * x, axis=-1, keepdims=True)  # (B,1)
        return tf.squeeze(q, axis=-1)             # 标量或 (B,)

    # ---- 方便：联合 / 个体基线都用同一个 Q ----
    def compute_joint_baseline(self, global_obs, pi_joint):
        return self.compute_q(global_obs, pi_joint)

    def compute_individual_baseline(self, global_obs, pi_mixed):
        return self.compute_q(global_obs, pi_mixed)

# ============================================================
# MACA2C Policy: Actor (multi-head) + MACA-style Q-critic
# ============================================================

class MACA2CPolicy(tf.keras.Model):
    """
    MACA2C: MA2C + MACA-style two-level credit assignment.
    - Actor: local obs -> LSTM -> multi-head logits
    - Critic: 使用 multi-agent Transformer encoder 对状态编码：
              global_obs (可以是 token 序列) -> Transformer -> z_s
              然后 Q(s, x) = v_phi(s) + w_phi(s)^T x, 线性依赖动作输入 x：
                * x = 联合 one-hot 动作向量  -> Q(s,a)
                * x = 联合策略概率向量 \bar{π} -> 联合基线 / 个体基线
    """

    def __init__(self,
                 obs_dim,
                 act_space_list,
                 global_obs_dim,
                 joint_action_dim,
                 n_lstm: int = 64,
                 name: str = "maca2c_policy",
                 # ===== critic Transformer 超参数（可选）=====
                 critic_d_model: int = 128,
                 critic_num_heads: int = 4,
                 critic_num_layers: int = 2,
                 critic_d_ff: int = 256):
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

        # ======== Critic: Transformer 状态编码 + 线性动作头 (MACA) ========
        # ---- Transformer encoder 配置 ----
        self.critic_d_model = critic_d_model
        self.critic_num_heads = critic_num_heads
        self.critic_num_layers = critic_num_layers
        self.critic_d_ff = critic_d_ff

        # 把输入的 global_obs (最后一维) 映射到 d_model 维 token
        # 若 global_obs 是 (B, gdim)，会先变成 (B, 1, gdim) 再投影；
        # 若已经是 (B, L, gdim)，则对每个 token 做投影。
        self.critic_input_proj = tf.keras.layers.Dense(
            self.critic_d_model,
            activation=None,
            name='critic_input_proj'
        )

        # 多层 Transformer Encoder blocks
        self.critic_blocks = [
            CriticTransformerBlock(
                d_model=self.critic_d_model,
                num_heads=self.critic_num_heads,
                d_ff=self.critic_d_ff,
                name=f'critic_block_{i}'
            )
            for i in range(self.critic_num_layers)
        ]

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

        # 只有在“在线决策”时更新内部状态。在线动作函数当前不在
        # GradientTape 内；这里停止梯度是防御性处理，不改变前向数值。
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
    # Critic: 状态编码（Transformer Encoder）
    # ----------------------------------------------------------------------
    def _encode_state(self, global_obs, training: bool = False):
        """
        将全局观测编码为状态 embedding z_s：
            - 如果 global_obs 形状是 (gdim,)：
                -> (1, gdim) -> (1, 1, gdim)
            - 如果形状是 (B, gdim)：
                -> (B, 1, gdim)
            - 如果形状是 (B, L, gdim)：
                -> 直接视为 L 个 token
        然后：
            1) 通过 critic_input_proj 投影到 d_model
            2) 堆叠若干 CriticTransformerBlock
            3) 对 token 维度做平均池化，得到 z_s: (B, d_model)
        """
        g = tf.convert_to_tensor(global_obs, dtype=tf.float32)

        if len(g.shape) == 2:
            # (B, gdim) -> (B, 1, gdim)，单 token 序列
            g = tf.expand_dims(g, axis=1)    # (B, 1, gdim)
        elif len(g.shape) != 3:
            raise ValueError(f"global_obs 必须是 (L, feat) 或 (B, L, feat)，当前形状为 {g.shape}")

        # 此时 g: (B, L, gdim)
        x = self.critic_input_proj(g)        # (B, L, d_model)

        # 堆叠多层 Transformer encoder
        for block in self.critic_blocks:
            x = block(x, training=training)  # (B, L, d_model)

        # 对 token 维度做平均池化，得到全局状态 embedding z_s
        z = tf.reduce_mean(x, axis=1)        # (B, d_model)
        return z

    # ----------------------------------------------------------------------
    # Critic: MACA-style Q(s, x) = v(s) + w(s)^T x
    # ----------------------------------------------------------------------
    def compute_q(self, global_obs, action_vec, training: bool = False):
        """
        global_obs:
            - (gdim,) 或 (B, gdim)
            - 也可以是 (B, L, gdim)，表示多 token（例如多智能体）的观测序列
        action_vec:
            - (action_dim,) 或 (B, action_dim)
            - 可以是:
                * 联合动作 one-hot 向量
                * 联合策略概率向量 \bar{π}
                * 混合向量（本 agent 用 prob，其它 agent 用 one-hot）
        return:
            Q(s, x)，标量或 (B,) 向量
        """
        x = tf.convert_to_tensor(action_vec, dtype=tf.float32)

        # 状态编码 z_s: (B, d_model)
        z = self._encode_state(global_obs, training=training)

        # v(s), w(s)
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
        self.coordination_contexts = []

    def add(self, obs, global_obs, acts, reward, done, mask=None,
            coordination_context=None):
        """
        obs:        (obs_dim,)
        global_obs: (global_obs_dim,) 或 以后可以扩展为 (num_tokens, feat_dim)
        acts:       list/array[int]，长度 = num_heads
        mask:       IAM 掩码（针对 head0 的动作维度），可以为 None
        """
        if coordination_context is None:
            raise ValueError("MACA2C requires a coordination context for every step.")
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.global_obs.append(np.asarray(global_obs, dtype=np.float32))
        self.acts.append(np.asarray(acts, dtype=np.int32))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.masks.append(mask)
        self.coordination_contexts.append(
            np.asarray(coordination_context, dtype=np.float32)
        )

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
                 args,
                 shared_critic=None,
                 shared_critic_opt=None,
                 coordination_weight_generator=None,
                 agent_type=None):
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
            - psi_joint（仅用于初始化 g_omega）
            - psi_ind（仅用于初始化 g_omega）
            - use_iam
            - max_grad_norm（可选）
            - （可选）maca2c_critic_d_model
            - （可选）maca2c_critic_num_heads
            - （可选）maca2c_critic_num_layers
            - （可选）maca2c_critic_d_ff
        """
        self.agent_index = agent_index
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.act_space_list = act_space_list
        self.num_heads = len(act_space_list)
        self.joint_action_dim = joint_action_dim
        if agent_type not in ("truck", "drone"):
            raise ValueError("agent_type must be either 'truck' or 'drone'.")
        if coordination_weight_generator is None:
            raise ValueError("A shared coordination weight generator is required.")
        self.agent_type = agent_type
        self.coordination_context_dim = 6 if agent_type == "truck" else 7
        self.coordination_weight_generator = coordination_weight_generator

        self.args = args
        self.gamma = float(getattr(args, 'gamma', 0.95))
        initial_psi_joint = float(getattr(args, 'psi_joint', 0.5))
        initial_psi_ind = float(getattr(args, 'psi_ind', 0.5))
        self.use_iam = bool(getattr(args, 'use_iam', False))
        self.max_grad_norm = getattr(args, 'max_grad_norm', None)
        # 熵系数：会在 train.py 中按步数通过 setattr(tr, "entropy_coef", ...) 动态覆盖，实现熵退火
        self.entropy_coef = float(getattr(args, 'entropy_coef', 1e-3))
        self.last_psi_ind = initial_psi_ind
        self.last_psi_joint = initial_psi_joint

        # critic Transformer 超参数（若未配置则使用默认值）
        critic_d_model = int(getattr(args, 'maca2c_critic_d_model', 128))
        critic_num_heads = int(getattr(args, 'maca2c_critic_num_heads', 4))
        critic_num_layers = int(getattr(args, 'maca2c_critic_num_layers', 2))
        critic_d_ff = int(getattr(args, 'maca2c_critic_d_ff', 256))

        # ===== Policy 网络（只用作 Actor）=====
        self.policy = MACA2CPolicy(
            obs_dim=obs_dim,
            act_space_list=act_space_list,
            global_obs_dim=global_obs_dim,
            joint_action_dim=joint_action_dim,
            n_lstm=getattr(args, 'num_units', 64),
            name=f'maca2c_policy_agent{agent_index}',
            critic_d_model=critic_d_model,        # 这些参数在 policy 里只会影响若你调用其 critic，
            critic_num_heads=critic_num_heads,    # 我们后面不再用 policy 的 critic，所以只是保留
            critic_num_layers=critic_num_layers,
            critic_d_ff=critic_d_ff
        )

        # 先跑一次 obs，确保 actor 部分建好
        dummy_obs = tf.zeros((1, 1, obs_dim), dtype=tf.float32)
        _logits, _ = self.policy(dummy_obs, training=False)

        # ===== 共享 Q-critic =====
        if shared_critic is not None:
            # 使用外部传进来的共享 critic
            self.critic = shared_critic
        else:
            # 若外部没传，就自己创建一份（单 agent 情况也能用）
            self.critic = MACA2CSharedCritic(
                global_obs_dim=global_obs_dim,
                joint_action_dim=joint_action_dim,
                critic_d_model=critic_d_model,
                critic_num_heads=critic_num_heads,
                critic_num_layers=critic_num_layers,
                critic_d_ff=critic_d_ff,
            )

        # dummy Q 前向，确保 critic 变量创建
        dummy_global = tf.zeros((1, 1, global_obs_dim), dtype=tf.float32)   # (B=1, L=1, feat)
        dummy_action_vec = tf.zeros((1, joint_action_dim), dtype=tf.float32)
        _ = self.critic.compute_q(dummy_global, dummy_action_vec)

        # # === Debug：打印当前 trainer 使用的 critic id ===
        # print(f"[MACA2C][Trainer init] agent {self.agent_index} uses critic id = {id(self.critic)}")

        # # === Debug：第一次构造 trainer 时，打印共享 critic 的变量信息 ===
        # if self.agent_index == 0:
        #     print("\n[MACA2C][Debug] Shared critic variables:")
        #     for v in self.critic.trainable_variables:
        #         print(f"  {v.name:40s} {v.shape}")
        # input("=== Debug：第一次构造 trainer 时，打印共享 critic 的变量信息 ===")

        # ===== 优化器 & 变量分组 =====
        lr_actor = float(getattr(args, 'lr_actor', getattr(args, 'lr', 1e-4)))
        lr_critic = float(getattr(args, 'lr_critic', getattr(args, 'lr', 3e-4)))

        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=lr_actor)

        if shared_critic_opt is not None:
            # 所有 agent 共用一个 critic 的 Adam
            self.critic_opt = shared_critic_opt
        else:
            self.critic_opt = tf.keras.optimizers.Adam(learning_rate=lr_critic)

        # 变量分组：
        #   Actor: 仍然从 policy 中取带 'actor_' 前缀的变量
        #   Critic: 来自共享 critic 实例
        self.actor_vars = [v for v in self.policy.trainable_variables if 'actor_' in v.name]
        self.critic_vars = self.critic.trainable_variables

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
    def experience(self, obs, global_obs, actions_per_head, reward, done, mask=None,
                   coordination_context=None):
        """
        存一条 (o_t, s_t, a_t, r_t, done_t, mask_t) 进本 agent 的 on-policy buffer
        actions_per_head: list[int], 与 action() 返回的第一项对应
        """
        if coordination_context is None:
            raise ValueError("coordination_context must not be None.")
        context = np.asarray(coordination_context, dtype=np.float32).reshape(-1)
        if context.size != self.coordination_context_dim:
            raise ValueError(
                f"{self.agent_type} agent {self.agent_index} requires a "
                f"{self.coordination_context_dim}-D coordination context; "
                f"got {context.size}."
            )
        self.buffer.add(
            obs, global_obs, actions_per_head, reward, done, mask,
            coordination_context=context
        )
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
        # [修改点 1]：正确处理 Global Obs 的维度
        # self.buffer.global_obs 是一个 list，每个元素形状为 (1, n_agents, gdim)
        # 使用 concatenate 在 axis=0 拼接，得到 (T, n_agents, gdim)
        # 这样 T 就被视为 Batch Size，n_agents 被视为 Sequence Length
        global_obs_batch = np.concatenate(self.buffer.global_obs, axis=0)  # (T, n_agents, gdim)

        acts_batch = np.stack(self.buffer.acts, axis=0)            # (T, num_heads)
        masks_batch = self.buffer.masks
        coordination_contexts = np.stack(
            self.buffer.coordination_contexts, axis=0
        )

        # if self.agent_index == 0 and T > 0:
        #     print("[MACA2C][Debug] global_obs_batch shape =", global_obs_batch.shape)
        #     print("[MACA2C][Debug] example s_t shape =", global_obs_batch[0].shape)

        # input()

        # # ===== 1) 计算 MC returns =====
        # returns = self.buffer.get_returns(final_value=0.0)         # (T,)
        # ===== 1) 计算带 bootstrap 的 MC returns =====
        # 如果这一段轨迹的最后一步已经终止（done=True），则 final_value=0
        # 否则，用共享 critic 估计 V(s_T) ≈ Q(s_T, π_joint_T) 作为 bootstrap
        last_done = self.buffer.dones[-1]
        if last_done:
            final_value = 0.0
        else:
            # 取最后一步的全局观测和联合策略向量
            s_T = global_obs_batch[-1]      # 形状: (n_agents, gdim)
            pi_joint_T = joint_policies[-1] # 形状: (joint_action_dim,)

            # b^{Jnt}(s_T) = Q(s_T, π_joint_T)
            v_T = self.critic.compute_joint_baseline(s_T, pi_joint_T)
            # 转成 Python float，兼容 Tensor / 标量两种返回形式
            final_value = float(v_T.numpy() if isinstance(v_T, tf.Tensor) else v_T)
        print(f"[MACA2C][agent {self.agent_index}] bootstrap final_value = {final_value:.4f}, last_done = {last_done}")
        # input()

        returns = self.buffer.get_returns(final_value=final_value)  # (T,)

        # ===== 2) 计算两级 MACA 优势 A_i(t) =====
        advantages = []
        psi_ind_values = []
        psi_joint_values = []

        for t in range(T):
            s_t = global_obs_batch[t]
            a_joint = joint_action_onehots[t]
            pi_joint = joint_policies[t]

            # Q(s_t, a_t)
            q_sa = self.critic.compute_q(s_t, a_joint)

            # 联合基线 b^{Jnt}(s_t) = Q(s_t, \bar{π}^{Jnt}_t)
            b_jnt = self.critic.compute_joint_baseline(s_t, pi_joint)

            # 个体基线 b^{Ind}_i(s_t,a_t) = Q(s_t, \bar{π}_{G={i}}(t))
            # 构造 \bar{π}_{G={i}}：自己用 prob，其它 agent 用 one-hot
            mixed_parts = []
            for j, (onehot_j, prob_j) in enumerate(zip(all_agents_onehots[t], all_agents_probs[t])):
                if j == self.agent_index:
                    mixed_parts.append(tf.convert_to_tensor(prob_j, dtype=tf.float32))
                else:
                    mixed_parts.append(tf.convert_to_tensor(onehot_j, dtype=tf.float32))
            pi_mixed = tf.concat(mixed_parts, axis=-1)

            b_ind = self.critic.compute_individual_baseline(s_t, pi_mixed)

            # g_omega returns [psi_ind, psi_joint]. It is evaluated outside
            # both actor and critic tapes and receives no backpropagated loss.
            z_s = tf.stop_gradient(self.critic.encode_state(s_t, training=False))
            psi = self.coordination_weight_generator.for_agent(
                self.agent_type, z_s, coordination_contexts[t]
            )
            psi = tf.squeeze(psi, axis=0)
            psi_ind = psi[0]
            psi_joint = psi[1]

            adv = q_sa - (psi_ind * b_ind + psi_joint * b_jnt)
            advantages.append(float(adv.numpy() if isinstance(adv, tf.Tensor) else adv))
            psi_ind_values.append(float(psi_ind.numpy()))
            psi_joint_values.append(float(psi_joint.numpy()))

        self.last_psi_ind = float(np.mean(psi_ind_values))
        self.last_psi_joint = float(np.mean(psi_joint_values))

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

                    # # ===== 训练阶段也应用 IAM 掩码（仅 head0）=====
                    # if self.use_iam and head_idx == 0:
                    #     masked_logits_per_t = []
                    #     invalid_pick_count = 0
                    #     for t_step in range(T):
                    #         step_mask = masks_batch[t_step]
                    #         if step_mask is not None:
                    #             m = tf.convert_to_tensor(step_mask, dtype=tf.bool)
                    #             if not tf.reduce_any(m):
                    #                 m = tf.ones_like(m, dtype=tf.bool)
                    #             # 若该步实际选择的动作被 mask 禁用，记一下（方便 debug）
                    #             a_idx = acts_tensor[t_step, head_idx]
                    #             if int(a_idx.numpy()) >= 0 and (not bool(m[a_idx].numpy())):
                    #                 invalid_pick_count += 1
                    #             masked = tf.where(
                    #                 m,
                    #                 logits[t_step],
                    #                 tf.fill(tf.shape(logits[t_step]), tf.constant(-1e9, tf.float32))
                    #             )
                    #         else:
                    #             masked = logits[t_step]
                    #         masked_logits_per_t.append(masked)
                    #     logits = tf.stack(masked_logits_per_t, axis=0)  # (T, A)
                    #     if invalid_pick_count > 0:
                    #         print(f"[MACA2C][agent {self.agent_index}] head0 masked-pick count: "
                    #               f"{invalid_pick_count}/{T}")

                    # 直接使用 logits 计算概率
                    # log π(a|s) for this head
                    logp_all = tf.nn.log_softmax(logits, axis=-1)  # (T, A)
                    
                    # [新增] 数值安全钳制：防止极低概率导致 -inf
                    # log(e^-10) approx -10, log(e^-20) approx -20. 
                    # 钳制到 -20 对应约 2e-9 的概率，足够覆盖正常策略范围
                    logp_all = tf.clip_by_value(logp_all, -20.0, 0.0)

                    idx = tf.stack(
                        [tf.range(T, dtype=tf.int32), acts_tensor[:, head_idx]],
                        axis=1
                    )
                    logp_head = tf.gather_nd(logp_all, idx)        # (T,)
                    total_logp += logp_head

                    # ===== 熵项 H[π]，和 MA2C 写法保持一致 =====
                    probs = tf.nn.softmax(logits, axis=-1)
                    # entropy = -tf.reduce_sum(probs * logp_all, axis=-1)   # (T,)
                    # entropy_terms.append(tf.reduce_mean(entropy))

                    # [新增] 熵计算同样使用受钳制的 logp_all，防止 NaN
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

                # [修改点 2]：批量计算 Q 值，避免循环（可选优化，提高效率）
                # 我们可以直接将整个 batch 传入 compute_q，利用并行计算
                # s_batch: (T, n_agents, gdim) -> 对应 compute_q 的 global_obs
                # a_batch: list of (joint_dim,) -> stack 为 (T, joint_dim)
                
                # 为了保持代码结构最小改动，我们先保持循环，但维度是正确的
                # s_t = global_obs_batch[t]  -> (n_agents, gdim)
                # 这会被 Critic 的 _encode_state 视为 (L, feat)，自动增加 Batch=1 -> (1, L, feat)
                # 计算出 (1,) 的 Q 值，这也是正确的。

                q_preds = []
                for t in range(T):
                    s_t = global_obs_batch[t]          # (n_agents, gdim)
                    a_joint = joint_action_onehots[t]  # (joint_dim,)
                    q_val = self.critic.compute_q(s_t, a_joint) # 返回 scalar 或 (1,)
                    q_preds.append(q_val)
                q_preds = tf.stack(q_preds)            # (T,)
                
                # 如果 q_preds 是 (T, 1)，需要 squeeze
                if len(q_preds.shape) > 1:
                    q_preds = tf.squeeze(q_preds)

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
