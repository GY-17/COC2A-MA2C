import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STANDARD_DATASET_PATH = os.path.join(_PROJECT_ROOT, 'singlecenter', 'singlecenter-101-n175.txt')

cfg = {
    # 环境相关的参数，一并传给 BuildEnvironment，再由它传到 BaseTempDatabase
    'environment': {
        # Training intentionally generates random node coordinates. Evaluation
        # entry points opt into the standard dataset below.
        'dataset_path': None,
        'evaluation_dataset_path': _STANDARD_DATASET_PATH,
        'max_steps': 800,          # 🌟 [新增]：强制最大物理推演步数限制，防止死循环
        'num_trucks':    5,
        'num_drones':    5,
        'num_depots':    1,
        'num_customers':  4,
        'WT_max':        1000.0,   # 卡车最大载重
        'WD_max':        80.0,    # 无人机最大载重
        'ct_cost':       0.83,     # 卡车单位时间成本
        'cd_cost':       0.21,     # 无人机单位时间成本
        'max_charge': 50,
        'truck_speed':   0.5,      # 30km/h（千米/小时） 0.5km/min
        'drone_speed':   1,      # 60km/h（千米/小时）  1km/min
        'horizon':       300,     # 分钟
        'area_size':     15.0,    # 千米
        'road_damage_ratio': 0.2,  # 20%的节点只能由无人机访问
        'lambda_var': 2, # 【新增】：控制动态订单集中度的泊松参数 (1=平稳, 2=中等, 3=高峰)
        # 新增动态节点配置
        'dynamic_nodes': {
            'dod': 0.3,  # Degree of Dynamism (0.2, 0.5, 0.7)
            'delta_t': 2,  # 检查频率 (分钟)
            'enable': True,  # 是否启用动态节点
            # 'release_frac': 0.4, # 释放窗口的上限占比：0~(release_frac*horizon)
            'release_min_frac': 0.2, # 释放窗口的占比：(release_min_frac*horizon)~(release_max_frac*horizon)
            'release_max_frac': 0.6,
            'deadline_slack': [5, 30]  # 截止期 = release_time + U(slack_lo, slack_hi
        },
    },
    
    # 和节点属性相关的默认常量
    'node': {
        'customer': {
            'deadline': 100.0,   # 默认截止时间 D_i
            'alpha':    0.8,     # 延误惩罚权重 α_i
            'beta':     30.0,    # 服务收益 β_i
        },
        'depot': {
            'deadline': 480,  # 仓库一般没有强制截止
            'alpha':    0.0,           # 不产生延误惩罚
            'beta':     0.0,           # 不产生服务收益
        }
    },
    
    'training': {
        'num_episodes': 1500,    # 总共跑多少个 episode 后停止
        'batch_size':   60,     # 新增：每次训练抽样的大小
        'update_freq':  60,         # 新增：每隔多少步更新一次
        'target_update_freq': 100, # 新增：target网络更新频率（每20步更新一次）
        'min_buffer_size': 3000,    # 3000 → 5000 这里！！！！！
        'updates_per_call': 1,      # 新增：每次触发做几次梯度步
        'warmup_steps': 3000,       # 新增：高探索预热 这里！！！！
        
        # 'entropy_coef': 0.01,       # 新增/覆盖
        'tau': 0.01,                 # 新增/覆盖 目标网络软更新       
        'lr_actor': 1e-4,      # 新增：Actor 学习率
        'lr_critic': 3e-4,     # 新增：Critic 学习率（略大一点更快学好V）

        # === Entropy annealing (new) ===
        'entropy_coef_start': 0.005,   # 训练早期: 鼓励探索
        'entropy_coef_end':   0.001,   # 训练后期: 更果断(低熵)
        'entropy_anneal_after':  4_000,   # 多少环境步之后开始退火
        'entropy_anneal_steps':  40_000,  # 退火时长(步数)，之后就固定为 end
        'entropy_schedule': "linear",     # "linear" | "cosine" | "exp
        
        # ===MA2C/MAPPO
        'rollout_len': 60,     # 新增：分段回合长度（32~128都可以试）MA2C
        'ppo_epochs': 5,             # 新增：每个rollout重复训练的epoch数 MAPPO
        'minibatch_size': 128,       # 新增：小批大小

        # ===COMADDPG ===
        'cf_K': 5,     #反事实基线采样次数 K
        'cf_use_target': True, #基线用 target critic（稳定）
        'baseline_masked': True,  #采样时对 head-0 应用 IAM 掩码

        # === MACA2C 超参数 ===
        # [原有参数，语义已调整] 以前是优势函数中的固定权重；现在仅用于初始化
        # g_omega 的输出偏置，使首次 CMA-ES 更新前 Softmax 输出为
        # [psi_ind, psi_joint] = [0.6, 0.4]。后续每个状态的权重由 g_omega 动态生成。
        'maca2c_psi_joint': 0.4,
        'maca2c_psi_ind': 0.6,

        # [本次新增] 卡车/无人机各自的双塔交互网络中，状态塔和上下文塔的
        # 输出维度。当前为 4，即 s_i、c_i 和 s_i*c_i 都是 4 维。
        # 增大该值会提高表达能力，同时会显著增加 CMA-ES 的搜索维度。
        'maca2c_coord_hidden_dim': 4,

        # === [本次新增] g_omega 的 CMA-ES 外层优化参数 ===
        # 是否启用动态权重的 CMA-ES 外层优化。False 时保持当前 omega，
        # 但仍通过 g_omega 计算权重。
        'maca2c_cma_enabled': True,
        # K：每完成多少个主训练 episode，执行一代 CMA-ES 优化。
        'maca2c_cma_interval': 100,
        # M：每代 ask() 生成的候选 omega 数量。候选越多，搜索更充分，耗时越高。
        'maca2c_cma_population': 5,
        # H：每个候选从同一份 actor/critic/Adam/随机状态快照出发，
        # 使用该候选 omega 进行多少个临时训练 episode。临时网络更新最终会被丢弃。
        'maca2c_cma_inner_episodes': 20,
        # 计算 R_before 和每个 R_after 时使用的固定种子评估 episode 数。
        # 当前一次外层触发共额外运行 5*20 + (5+1)*20 = 220 个 episode。
        'maca2c_cma_eval_episodes': 20,
        # CMA-ES 初始搜索标准差 sigma。越大，候选 omega 离当前搜索中心越远。
        'maca2c_cma_sigma': 0.05,
        # CMA-ES 自身的候选采样种子，用于复现 ask() 生成的候选序列。
        'maca2c_cma_seed': 2025,
        # 策略评估的基础种子；同一代的 R_before 与全部 R_after 使用同一组种子。
        'maca2c_cma_eval_seed': 31000,

        # [原有参数] Actor LSTM 的隐藏单元数。
        'maca2c_lstm_units': 64,
        # [原有参数] Actor/Critic 的梯度范数裁剪上限；None 表示不裁剪。
        'maca2c_max_grad_norm': 0.5,

        # === [原有参数] MACA2C Transformer Critic 结构 ===
        # 每个智能体观测 token 投影后的维度，也是传入 g_omega 的 z_s 维度。
        'maca2c_critic_d_model': 128,
        # 多头注意力的头数；d_model 必须可以被 num_heads 整除。
        'maca2c_critic_num_heads': 4,
        # Transformer encoder block 的堆叠层数。
        'maca2c_critic_num_layers': 2,
        # 每个 Transformer block 内前馈网络的隐藏维度。
        'maca2c_critic_d_ff': 256,
        
        # === 新增：reward 中成本项的权重 ===
        'reward_travel_weight': 2.0,  # 行驶成本在 reward 里的权重（你可以试 1.5 / 2.0）
        'reward_delay_weight': 1.0,   # 延期惩罚在 reward 里的权重
        # 只缩放送入算法的即时 reward；不会缩放 TensorBoard 的原始 Total_Cost。
        'reward_scale': 20.0,

    }
}
