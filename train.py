import argparse
import os
import tensorflow as tf
import time
import gc  # <--- 【新增】引入垃圾回收模块
from datetime import datetime
from trucks_and_drones.build_env import BuildEnvironment
import maddpg.trainer.utils as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from tensorflow.keras import layers
from trucks_and_drones.config import cfg
import numpy as np
from maddpg.trainer.cmaes_weight_optimizer import (
    MACA2CTrainingSnapshot,
    capture_random_state,
    restore_random_state,
    seed_python_and_numpy,
)

def parse_args():
    parser = argparse.ArgumentParser("Multi-Agent RL experiments")

    # ========== 新增:算法选择参数 ==========
    parser.add_argument("--algorithm", type=str, default="maddpg",
                       choices=["maddpg", "iql", "ma2c", "comaddpg", "cima","mappo", "maca2c"],
                       help="Algorithm to use: maddpg/iql/ma2c/comaddpg/cima/mappo/maca2c")

    # 在parse_args()中添加MAPPO参数
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="PPO clip parameter epsilon")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda parameter")
    parser.add_argument("--value_coef", type=float, default=0.5,
                        help="Value loss coefficient")

    # Environment parameters
    parser.add_argument("--env-config", type=str, default=None,
                        help="Path to Python script that configures BuildEnvironment")
    parser.add_argument("--max-episode-len", type=int, default=100,
                        help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000,
                        help="number of episodes")
    # Core training parameters
    parser.add_argument("--lr_actor", type=float, default=1e-4,
                        help="learning rate for actor")
    parser.add_argument("--lr_critic", type=float, default=3e-4,
                        help="learning rate for critic")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")
    parser.add_argument("--batch-size", type=int,
                        help="batch size")
    parser.add_argument("--num-units", type=int, default=64,
                        help="number of units in the MLP layers")
    parser.add_argument("--update-freq", type=int,
                        help="how often (in training steps) to update networks")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="experiment",
                        help="name of the experiment (used for saving)")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/",
                        help="directory in which to save models")
    parser.add_argument("--save-rate", type=int, default=2,
                        help="save model once every this many episodes")

    # TensorBoard logging
    parser.add_argument("--log-dir", type=str, default="./logs/",
                        help="directory for TensorBoard logs")

    # Flags for PER and IAM
    parser.add_argument("--use-per", action="store_true", default=False,
                        help="use prioritized experience replay")
    parser.add_argument("--use-iam", action="store_true", default=False,
                        help="use invalid action masking")

    # 添加可视化控制参数
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Enable visualization during training")
    parser.add_argument("--visualize-interval", type=int, default=1,
                        help="Visualize every N episodes (only when --visualize is enabled)")
    

    # PER parameters
    parser.add_argument("--per_mu", type=float, default=0.6,
                        help="Priority exponent mu for PER (Equation 33)")
    parser.add_argument("--per_sigma", type=float, default=1.0,
                        help="Sampling probability exponent sigma for PER (Equation 34)")
    parser.add_argument("--per_eta", type=float, default=1e-6,
                        help="Small constant eta to avoid zero priority")
    parser.add_argument("--per_beta", type=float, default=0.4,
                        help="Importance sampling exponent beta for PER")

    parser.add_argument("--hidden_units_actor", nargs='+', type=int, default=[512, 256],
                        help="Hidden layer sizes for actor network")
    parser.add_argument("--hidden_units_critic", nargs='+', type=int, default=[1024, 512, 256],
                        help="Hidden layer sizes for critic network")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function: relu or tanh")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon value for epsilon-greedy exploration")

    return parser.parse_args()


def mlp_actor(input, num_outputs, scope, reuse=False):
    x = layers.Dense(512, activation='relu')(input)
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(num_outputs)(x)
    return out


def mlp_critic(input, num_outputs, scope, reuse=False):
    x = layers.Dense(1024, activation='relu')(input)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(num_outputs)(x)
    return out

# 线性 ε 调度：warmup 步内为 1.0，之后在 decay_steps 内下降到 0.1
def get_epsilon(step, warmup_steps=1500, eps_max=1.0, eps_min=0.02, decay_steps=40000):
    if step < warmup_steps:
        return eps_max
    # ratio = min(1.0, max(0.0, (step - warmup_steps) / float(decay_steps)))
    # return eps_max + (eps_min - eps_max) * ratio
    # ⭐ 使用指数衰减替代线性衰减
    decay_ratio = (step - warmup_steps) / float(decay_steps)
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    return eps_min + (eps_max - eps_min) * np.exp(-3 * decay_ratio)  # 指数衰减

def get_entropy_coef(step, start=1e-2, end=5e-4, anneal_after=5000, anneal_steps=40000, schedule="linear"):
    """根据训练步数返回当前熵系数。"""
    if step < anneal_after:
        return float(start)
    t = (step - anneal_after) / float(max(1, anneal_steps))
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)

    if schedule == "linear":
        return float(start + (end - start) * t)
    elif schedule == "cosine":
        # 余弦退火：开始高，平滑下降到 end
        return float(end + 0.5 * (start - end) * (1 + np.cos(np.pi * t)))
    elif schedule == "exp":
        # 指数退火：由 start 指数下降到 end
        k = np.log(max(1e-12, end) / max(1e-12, start)) / max(1, anneal_steps)
        return float(start * np.exp(k * (step - anneal_after)))
    else:
        return float(start + (end - start) * t)


def _multihead_onehot_from_ids(action_ids, head_dims):
    onehots = []
    for action, dimension in zip(action_ids, head_dims):
        onehot = np.zeros(dimension, dtype=np.float32)
        action = int(action) if action is not None else 0
        action = int(np.clip(action, 0, dimension - 1))
        onehot[action] = 1.0
        onehots.append(onehot)
    return np.concatenate(onehots, axis=-1).astype(np.float32)


def _set_maca2c_entropy(trainers, args, train_step):
    entropy = get_entropy_coef(
        step=train_step,
        start=getattr(args, "entropy_coef_start", 1e-2),
        end=getattr(args, "entropy_coef_end", 5e-4),
        anneal_after=getattr(args, "entropy_anneal_after", 5000),
        anneal_steps=getattr(args, "entropy_anneal_steps", 40000),
        schedule=getattr(args, "entropy_schedule", "linear"),
    )
    for trainer in trainers:
        trainer.entropy_coef = float(entropy)
    return float(entropy)


def _update_maca2c_segment(trainers, joint_onehots, joint_policies,
                           all_onehots, all_probs):
    losses = []
    for trainer in trainers:
        losses.append(trainer.update(
            joint_action_onehots=joint_onehots,
            joint_policies=joint_policies,
            all_agents_onehots=all_onehots,
            all_agents_probs=all_probs
        ))
    return losses


def _reset_maca2c_runtime(trainers):
    for trainer in trainers:
        trainer.buffer.reset()
        trainer.policy.reset_states()


def _cleanup_maca2c_runtime(trainers):
    """Release short-lived rollout objects after isolated CMA-ES runs."""
    _reset_maca2c_runtime(trainers)
    # The reset above drops the model-held state; collect Python containers
    # that became unreachable while the candidate was being evaluated.
    gc.collect()


def _process_rss_mb():
    """Return current process RSS in MiB without adding a hard dependency."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 2)
    except Exception:
        # Linux fallback used on the training server. VmRSS is in kB.
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
    return None


def _log_process_rss(stage):
    rss_mb = _process_rss_mb()
    if rss_mb is not None:
        print(f"[MEM] {stage}: RSS={rss_mb:.1f} MiB", flush=True)


def _run_maca2c_episode(env, trainers, args, head_dims_per_agent,
                         train_step, train_policy):
    """Run one isolated MACA2C episode for CMA inner training/evaluation."""
    _reset_maca2c_runtime(trainers)
    obs_n, _ = env.reset()
    n_agents = len(trainers)
    total_reward = 0.0

    joint_onehots = []
    joint_policies = []
    all_onehots = []
    all_probs = []

    done = False
    episode_steps = 0
    max_steps = max(1, int(getattr(
        env, 'max_steps', getattr(env.temp_db, 'max_steps', 1000)
    )))
    while not done:
        # These values describe s_t and must be captured before env.step().
        contexts = env.obs_encoder.get_coordination_contexts()
        if len(contexts) != n_agents:
            raise RuntimeError(
                f"Expected {n_agents} coordination contexts, got {len(contexts)}."
            )
        masks = [env.get_mask(agent_index=i) for i in range(n_agents)] \
            if args.use_iam else [None] * n_agents
        if train_policy:
            _set_maca2c_entropy(trainers, args, train_step)

        actions = []
        step_probs = []
        for index, trainer in enumerate(trainers):
            actions_i, _, probs_i = trainer.action(obs_n[index], masks[index])
            actions.append(actions_i)
            step_probs.append(probs_i)

        (next_obs_n, _), rewards, done, info = env.step(actions)
        episode_steps += 1
        # Defensive fallback for environments constructed outside CustomEnv.
        if not done and episode_steps >= max_steps:
            done = True
            print(
                f"[MACA2C] inner episode force-terminated at max_steps={max_steps}",
                flush=True
            )
        if isinstance(info, dict) and info.get("agent_step_rewards") is not None:
            rewards = info["agent_step_rewards"]

        executed_actions = info.get("executed_actions_multihead") \
            if isinstance(info, dict) else None
        if executed_actions is None:
            executed_actions = getattr(
                env.temp_db, "last_executed_actions_multihead", None
            )
        if executed_actions is None:
            executed_actions = actions

        team_reward = float(np.sum(rewards))
        total_reward += team_reward

        if train_policy:
            step_onehots = [
                _multihead_onehot_from_ids(
                    executed_actions[i], head_dims_per_agent[i]
                )
                for i in range(n_agents)
            ]
            joint_onehots.append(
                np.concatenate(step_onehots, axis=-1).astype(np.float32)
            )
            joint_policies.append(
                np.concatenate(step_probs, axis=-1).astype(np.float32)
            )
            all_onehots.append(step_onehots)
            all_probs.append(step_probs)

            obs_tokens, _ = pad1d_list(obs_n)
            global_tokens = obs_tokens[np.newaxis, :, :].astype(np.float32)
            for i, trainer in enumerate(trainers):
                trainer.experience(
                    obs=obs_n[i],
                    global_obs=global_tokens,
                    actions_per_head=executed_actions[i],
                    reward=team_reward,
                    done=done,
                    mask=masks[i],
                    coordination_context=contexts[i]
                )

            rollout_len = getattr(args, "rollout_len", None)
            if rollout_len is not None and rollout_len > 0 and \
                    len(trainers[0].buffer) >= rollout_len:
                _update_maca2c_segment(
                    trainers, joint_onehots, joint_policies,
                    all_onehots, all_probs
                )
                joint_onehots = []
                joint_policies = []
                all_onehots = []
                all_probs = []

        obs_n = next_obs_n
        train_step += 1

    if train_policy and len(trainers[0].buffer) > 0:
        _update_maca2c_segment(
            trainers, joint_onehots, joint_policies, all_onehots, all_probs
        )
    _reset_maca2c_runtime(trainers)
    return total_reward, train_step


def _evaluate_maca2c_policy(env, trainers, args, head_dims_per_agent,
                            train_step, evaluation_seeds, progress_label='evaluation'):
    caller_random_state = capture_random_state()
    rewards = []
    try:
        total_evaluations = len(evaluation_seeds)
        for evaluation_index, seed in enumerate(evaluation_seeds, start=1):
            seed_python_and_numpy(seed)
            reward, _ = _run_maca2c_episode(
                env=env, trainers=trainers, args=args,
                head_dims_per_agent=head_dims_per_agent,
                train_step=train_step, train_policy=False
            )
            rewards.append(reward)
            if evaluation_index % 5 == 0 or evaluation_index == total_evaluations:
                print(
                    f"[CMA-ES] {progress_label}: "
                    f"{evaluation_index}/{total_evaluations} episodes complete",
                    flush=True
                )
    finally:
        restore_random_state(caller_random_state)
        _cleanup_maca2c_runtime(trainers)
    return float(np.mean(rewards))


def _optimize_maca2c_weights(env, trainers, args, head_dims_per_agent,
                              weight_generator, cma_optimizer, train_step,
                              completed_episodes):
    if any(len(trainer.buffer) for trainer in trainers):
        raise RuntimeError(
            "CMA-ES may only run at an episode boundary with empty rollouts."
        )

    snapshot = MACA2CTrainingSnapshot.capture(trainers)
    original_omega = weight_generator.flatten_parameters()
    original_env_episode = getattr(env, "count_episodes", 0)
    eval_seed_base = int(args.maca2c_cma_eval_seed) + (
        int(cma_optimizer.generation) * 100000
    )
    evaluation_seeds = [
        eval_seed_base + i for i in range(int(args.maca2c_cma_eval_episodes))
    ]

    best_omega = original_omega
    result = None
    try:
        env.count_episodes = original_env_episode
        _log_process_rss("CMA-ES start")
        reward_before = _evaluate_maca2c_policy(
            env, trainers, args, head_dims_per_agent, train_step,
            evaluation_seeds, progress_label='baseline evaluation'
        )
        candidates = cma_optimizer.ask()
        fitnesses = []
        gains = []

        for candidate_index, candidate in enumerate(candidates):
            snapshot.restore(trainers, restore_rng=True)
            env.count_episodes = original_env_episode
            weight_generator.restore_parameters(candidate)

            candidate_train_step = train_step
            inner_episodes = int(args.maca2c_cma_inner_episodes)
            for inner_index in range(inner_episodes):
                _, candidate_train_step = _run_maca2c_episode(
                    env=env, trainers=trainers, args=args,
                    head_dims_per_agent=head_dims_per_agent,
                    train_step=candidate_train_step, train_policy=True
                )
                if ((inner_index + 1) % 5 == 0 or
                        inner_index + 1 == inner_episodes):
                    print(
                        f"[CMA-ES] candidate {candidate_index + 1}/{len(candidates)} "
                        f"inner episode {inner_index + 1}/{inner_episodes} complete",
                        flush=True
                    )
                    _cleanup_maca2c_runtime(trainers)
                    _log_process_rss(
                        f"candidate {candidate_index + 1} inner {inner_index + 1}"
                    )

            env.count_episodes = original_env_episode
            reward_after = _evaluate_maca2c_policy(
                env, trainers, args, head_dims_per_agent,
                candidate_train_step, evaluation_seeds,
                progress_label=f'candidate {candidate_index + 1} evaluation'
            )
            gain = float(reward_after - reward_before)
            gains.append(gain)
            fitnesses.append(-gain)
            print(
                f"[CMA-ES] candidate {candidate_index + 1}/{len(candidates)}: "
                f"R_before={reward_before:.4f}, R_after={reward_after:.4f}, "
                f"gain={gain:.4f}"
            )
            _cleanup_maca2c_runtime(trainers)
            _log_process_rss(f"candidate {candidate_index + 1} complete")

        best_omega = cma_optimizer.tell(candidates, fitnesses)
        result = {
            "reward_before": reward_before,
            "best_gain": float(np.max(gains)),
            "mean_gain": float(np.mean(gains)),
            "generation": int(cma_optimizer.generation),
            "completed_episodes": int(completed_episodes),
        }
    finally:
        # Candidate actor/critic updates are deliberately discarded.
        snapshot.restore(trainers, restore_rng=True)
        env.count_episodes = original_env_episode
        weight_generator.restore_parameters(best_omega if result is not None else original_omega)
        _cleanup_maca2c_runtime(trainers)
        _log_process_rss("CMA-ES cleanup")

    return result

def pad1d_list(xs, fill=0.0):
    """
    把一个 list 里不同长度的 1D 向量补齐到相同长度并 stack 成 2D 数组：
        xs: list[np.ndarray(?,)], 每个元素是一维观测
    返回:
        stacked: (len(xs), max_len)
        max_len: int
    """
    arrs = [np.asarray(x, dtype=np.float32).reshape(-1) for x in xs]
    maxlen = max(a.shape[0] for a in arrs)
    out = [
        np.pad(a, (0, maxlen - a.shape[0]),
               mode="constant", constant_values=fill)
        for a in arrs
    ]
    return np.stack(out, axis=0), maxlen

def train(args):
    # ========== 动态导入算法 ==========
    if args.algorithm == "maddpg":
        from maddpg.trainer.maddpg import MADDPGAgentTrainer as TrainerClass
    elif args.algorithm == "ma2c":
        from maddpg.trainer.ma2c import MA2CAgentTrainer as TrainerClass
    elif args.algorithm == "mappo":
        from maddpg.trainer.mappo import MAPPOAgentTrainer as TrainerClass
    elif args.algorithm == "comaddpg":
        from maddpg.trainer.comaddpg import COMADDPGAgentTrainer as TrainerClass
    elif args.algorithm == "cima":
        from maddpg.trainer.cima import CIMAAgentTrainer as TrainerClass
    elif args.algorithm == "maca2c":
        # MACA2C: MA2C + MACA 多级显示信用分配
        from maddpg.trainer.maca2c import MACA2CAgentTrainer, MACA2CSharedCritic
        from maddpg.trainer.coordination_weight import CoordinationWeightGenerator
        from maddpg.trainer.cmaes_weight_optimizer import CMAESWeightOptimizer
        TrainerClass = MACA2CAgentTrainer

    print(f"🤖 Using algorithm: {args.algorithm.upper()}")
    
    # 设置TensorBoard日志记录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.exp_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建TensorBoard writers
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    episode_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'episode'))
    
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    print(f"📈 Run 'tensorboard --logdir {log_dir}' to view training curves")
    
    # 覆盖命令行参数：读取 config 中的训练轮数
    args.num_episodes = cfg['training']['num_episodes']
    args.batch_size   = cfg['training']['batch_size']
    args.update_freq  = cfg['training']['update_freq']
    args.target_update_freq = cfg['training']['target_update_freq']
    args.min_buffer_size = cfg['training']['min_buffer_size']
    args.updates_per_call   = cfg['training'].get('updates_per_call')
    args.warmup_steps       = cfg['training'].get('warmup_steps')
    # args.entropy_coef       = cfg['training'].get('entropy_coef')
    args.tau                = cfg['training'].get('tau')
    args.lr_actor   = cfg['training'].get('lr_actor')
    args.lr_critic  = cfg['training'].get('lr_critic')
    args.rollout_len = cfg['training'].get('rollout_len')

    # === 从 config 读取熵退火超参（若未配置则给默认值）===
    args.entropy_coef_start   = cfg['training'].get('entropy_coef_start', 1e-2)
    args.entropy_coef_end     = cfg['training'].get('entropy_coef_end',   5e-4)
    args.entropy_anneal_after = cfg['training'].get('entropy_anneal_after', 5000)
    args.entropy_anneal_steps = cfg['training'].get('entropy_anneal_steps', 40000)
    args.entropy_schedule     = cfg['training'].get('entropy_schedule', 'linear')

    #=== 从config中读取反事实comaddpg相关参数 ===
    args.cf_K = cfg['training'].get('cf_K', 8)                 # 反事实基线采样次数 K
    args.cf_use_target = cfg['training'].get('cf_use_target', True)  # 基线用 target critic（稳定）
    args.baseline_masked = cfg['training'].get('baseline_masked', True)  # 采样时对 head-0 应用 IAM 掩码
    args.entropy_coef = cfg['training'].get('entropy_coef', 1e-3)       # 熵正则权重

    # === MACA2C 专用超参数（若 cfg 中未设置则回落到默认）===
    args.psi_joint = cfg['training'].get('maca2c_psi_joint', getattr(args, 'psi_joint', 0.5))
    args.psi_ind   = cfg['training'].get('maca2c_psi_ind',   getattr(args, 'psi_ind',   0.5))
    args.maca2c_cma_enabled = cfg['training'].get('maca2c_cma_enabled', True)
    args.maca2c_cma_interval = cfg['training'].get('maca2c_cma_interval', 100)
    args.maca2c_cma_population = cfg['training'].get('maca2c_cma_population', 5)
    args.maca2c_cma_inner_episodes = cfg['training'].get('maca2c_cma_inner_episodes', 20)
    args.maca2c_cma_eval_episodes = cfg['training'].get('maca2c_cma_eval_episodes', 20)
    args.maca2c_cma_sigma = cfg['training'].get('maca2c_cma_sigma', 0.05)
    args.maca2c_cma_seed = cfg['training'].get('maca2c_cma_seed', 2025)
    args.maca2c_cma_eval_seed = cfg['training'].get('maca2c_cma_eval_seed', 31000)
    if args.algorithm.lower() == "maca2c" and args.maca2c_cma_enabled:
        positive_integer_fields = (
            "maca2c_cma_interval",
            "maca2c_cma_population",
            "maca2c_cma_inner_episodes",
            "maca2c_cma_eval_episodes",
        )
        for field in positive_integer_fields:
            if int(getattr(args, field)) <= 0:
                raise ValueError(f"{field} must be a positive integer.")
        if float(args.maca2c_cma_sigma) <= 0.0:
            raise ValueError("maca2c_cma_sigma must be positive.")
    args.max_grad_norm = cfg['training'].get(
        'maca2c_max_grad_norm',
        getattr(args, 'max_grad_norm', None)
    )
    maca2c_units = cfg['training'].get('maca2c_lstm_units', None)
    if maca2c_units is not None:
        # 统一用 args.num_units 传给 MACA2CPolicy 的 n_lstm
        args.num_units = maca2c_units

    # 1) 从 cfg 里读取环境配置
    env_cfg = cfg.get('environment', {})
    num_trucks    = env_cfg.get('num_trucks')
    num_drones    = env_cfg.get('num_drones')
    num_depots    = env_cfg.get('num_depots')
    num_customers = env_cfg.get('num_customers')


    # 2) 确定观测字段：用 temp_db 里实际有的 'n_items' 而不是 'demand'
    obs_contin   = [ 'TW', 'time', 'n_items', 'deadline']
    obs_discrete = ['ET', 'ED', 'NT', 'ND', 'truck_node', 'drone_node', 'delta']
    # 'delta'是“unassigned”的底层字段
    tandem_fields = obs_discrete + obs_contin + ['LT_time', 'LD_time']


    # —— 3) 一次性构造环境 ——
    builder = (
        BuildEnvironment(
            "VRPD",
            enable_visualization=args.visualize  # 传递可视化开关
        )
        .trucks(num=num_trucks)
        .drones(num=num_drones)
        .depots(num=num_depots)
        .customers(num=num_customers)
        .visuals()
        .observations(
            contin_inputs=obs_contin,
            discrete_inputs=obs_discrete,  # 用 NT/ND
            combine_per_index=[tandem_fields for _ in range(num_trucks)],
            combine_per_type=['vehicles', 'customers', 'depots'],  # 关键：全局观测来源
            flatten=True,  # 把每个 agent 的观测扁平化
            output_as_array=False, # 返回 list_of_agent_obs 而不是大向量
        )
        .actions()
        .rewards()
        .compile()

    )
    env = builder.build()
    
    # 4) 多智能体训练准备
    n_agents = len(builder.trucks) + len(builder.drones)
    
    # obs_shape_n = [env.observation_space[i].shape for i in range(n_agents)]
    obs_n, global_obs = env.reset()
    obs_shape_n = [obs.shape for obs in obs_n]
    
    # 获取全局观测维度
    obs_n, global_obs = env.reset()
    obs_shape_n = [obs.shape for obs in obs_n]
    
    # 获取全局观测维度（扁平版，供大多数算法使用）
    global_obs_dim_flat = global_obs.shape[0] if hasattr(global_obs, 'shape') else len(global_obs)

    # ===== MACA2C 专用：基于所有智能体局部观测构造 multi-token 形式 =====
    if args.algorithm.lower() == "maca2c":
        # 把每个 agent 的 1D obs 补齐后 stack 成 (n_agents, token_dim)
        maca2c_obs_tokens0, maca2c_token_dim = pad1d_list(obs_n)
        # 对于 MACA2C，global_obs_dim 表示“每个 token 的特征维度”
        global_obs_dim = maca2c_token_dim
    else:
        # 其它算法保持原来的扁平全局观测维度
        global_obs_dim = global_obs_dim_flat

    # print(f"🌍 Global observation dimension (flat): {global_obs_dim_flat}")
    # if args.algorithm.lower() == "maca2c":
    #     print(f"🌍 MACA2C token feature dimension: {global_obs_dim}")

    # print(">>> obs_shape_n =", obs_shape_n)
    # # 详细分析观测空间组成
    # print(f"\n🔬 === 观测空间组成分析 ===")
    
    # 从temp_db获取详细信息进行调试
    temp_db = builder.temp_db
    # print(f"卡车状态 (ET): {temp_db.status_dict['ET']}")
    # print(f"无人机状态 (ED): {temp_db.status_dict['ED']}")
    # print(f"卡车载重 (TW): {temp_db.status_dict['TW']}")
    # print(f"无人机载重 (DW): {temp_db.status_dict['DW']}")
    # print(f"节点需求 (n_items): {temp_db.status_dict['n_items']}")
    # print(f"节点截止时间 (deadline): {temp_db.constants_dict.get('deadline', 'Not found')}")
    # print(f"未分配状态 (delta): {temp_db.status_dict['delta']}")


    act_space_n = list(env.action_space.spaces)
    
    # print("Observation space:", env.observation_space)
    # print("Type:", type(env.observation_space))
    # for i, space in enumerate(env.observation_space):
    #     print(f" - Obs {i}: shape = {space.shape}, low = {space.low}, high = {space.high}")

    raw_aspace = env.act_decoder.action_space()
    # 原来 raw_aspace 是 Tuple(Tuple(Discrete,…), …)，先 unpack成 act_space_list：
    act_space_list = raw_aspace.spaces[0].spaces  # 取第一个 agent 的 “tuple of Discrete”    

    # ===== 仅 MACA2C 使用：每个智能体的 head 维度 + 联合动作向量总维度 =====
    if args.algorithm.lower() == "maca2c":
        # raw_aspace.spaces: 每个元素是一个 agent 的 Tuple(Discrete, Discrete, ...)
        head_dims_per_agent = [
            [space.n for space in per_agent_tuple.spaces]
            for per_agent_tuple in raw_aspace.spaces
        ]
        joint_action_dim = int(sum(sum(hs) for hs in head_dims_per_agent))
    else:
        head_dims_per_agent = None
        joint_action_dim = None

    trainers = []
    coordination_weight_generator = None
    cma_weight_optimizer = None
    if args.algorithm.lower() == "maca2c":
        # ===== 先创建一份共享 critic + 共享 Adam 优化器 =====
        shared_critic = MACA2CSharedCritic(
            global_obs_dim=global_obs_dim,
            joint_action_dim=joint_action_dim,
            critic_d_model=cfg['training'].get('maca2c_critic_d_model', 128),
            critic_num_heads=cfg['training'].get('maca2c_critic_num_heads', 4),
            critic_num_layers=cfg['training'].get('maca2c_critic_num_layers', 2),
            critic_d_ff=cfg['training'].get('maca2c_critic_d_ff', 256),
        )
        shared_critic_opt = tf.keras.optimizers.Adam(
            learning_rate=args.lr_critic
        )
        coordination_weight_generator = CoordinationWeightGenerator(
            state_dim=cfg['training'].get('maca2c_critic_d_model', 128),
            hidden_dim=cfg['training'].get('maca2c_coord_hidden_dim', 4),
            initial_individual_weight=args.psi_ind,
            initial_joint_weight=args.psi_joint,
        )
        if args.maca2c_cma_enabled:
            cma_weight_optimizer = CMAESWeightOptimizer(
                initial_parameters=coordination_weight_generator.flatten_parameters(),
                sigma=args.maca2c_cma_sigma,
                population_size=args.maca2c_cma_population,
                seed=args.maca2c_cma_seed,
            )
        print(
            f"[MACA2C] coordination omega dimension: "
            f"{coordination_weight_generator.parameter_count}"
        )

        # ===== 每个智能体各自一套 Actor + 共享 Critic =====
        for i in range(n_agents):
            per_agent_act_space_list = raw_aspace.spaces[i].spaces  # 该 agent 的多头 Discrete
            # obs_shape_n[i] 一般是形如 (obs_dim,) 的 tuple
            obs_dim_i = int(obs_shape_n[i][0]) if hasattr(obs_shape_n[i], '__len__') else int(obs_shape_n[i])
            trainers.append(
                TrainerClass(
                    agent_index=i,
                    obs_dim=obs_dim_i,
                    global_obs_dim=global_obs_dim,
                    act_space_list=per_agent_act_space_list,
                    joint_action_dim=joint_action_dim,
                    args=args,
                    shared_critic=shared_critic,
                    shared_critic_opt=shared_critic_opt,
                    coordination_weight_generator=coordination_weight_generator,
                    agent_type="truck" if i < num_trucks else "drone"
                )
            )
    else:
        # 原有算法保持不动
        for i in range(n_agents):
            trainers.append(
                TrainerClass(
                    name=f"agent_{i}",
                    obs_shape_n=obs_shape_n,
                    act_space_list=act_space_list,
                    agent_index=i,
                    args=args,
                    global_obs_dim=global_obs_dim,
                    # 添加缺失的环境参数
                    num_trucks=num_trucks,
                    num_drones=num_drones,
                    total_nodes=num_depots + num_customers
                )
            )

    # 5) 创建保存目录并保存初始模型
    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    train_step = 0
    
    # 6) 训练循环
    episode_rewards = [0.0]
    episode_step = 0
    obs_n, global_obs = env.reset()
    
    # 新增：专用“更新计数器”，让loss曲线连续
    update_steps = 0

    # ========== 新增：初始化总步数计数器 ==========
    env.visualizer.total_training_steps = 0
    # ==========================================

    # 初始化可视化 - 显示初始状态
    if args.visualize:
        print("🎬 Starting visualization...")
        env.visualizer.visualize_step(
            episode=len(episode_rewards)-1,
            step=episode_step,
            slow_down_pls=True,
            last_actions=None,
            last_rewards=None
        )
    print(f"Starting training with PER={args.use_per}, IAM={args.use_iam}")
    # 存储最后的动作用于显示
    last_actions = []
    last_rewards = []

    # ===== 仅 MACA2C：存一整集里联合动作/策略等 =====
    if args.algorithm.lower() == "maca2c":
        maca2c_joint_onehots = []    # [np.array(joint_action_dim,)] * T
        maca2c_joint_policies = []   # [np.array(joint_action_dim,)] * T
        maca2c_all_onehots = []      # [[agent0_onehot, agent1_onehot, ...], ...]
        maca2c_all_probs = []        # [[agent0_pi,     agent1_pi,     ...], ...]
    
    # 用于记录训练指标
    total_critic_loss = 0.0
    total_actor_loss = 0.0
    num_updates = 0

    while True:
        # # 生成动作掩码
        # mask_n = [env.get_mask() for _ in range(n_agents)] if args.use_iam else [None] * n_agents

        mask_n = [env.get_mask(agent_index=i) for i in range(n_agents)] if args.use_iam else [None] * n_agents

        # === ε 探索（使用 warmup_steps）===
        ws = args.warmup_steps if args.warmup_steps is not None else 0
        cur_eps = get_epsilon(train_step, warmup_steps=ws, eps_max=1.0, eps_min=0.02, decay_steps=40000)
        # for tr in trainers:
        #     tr.actor.epsilon = cur_eps
        for tr in trainers:
            # ✅ 只有MADDPG有epsilon属性，MA2C没有就跳过
            if hasattr(tr, 'actor') and hasattr(tr.actor, 'epsilon'):
                tr.actor.epsilon = cur_eps
        
        # --- 每个环境步 / 每次 update 之前：计算并下发“当前熵系数” ---
        cur_entropy = get_entropy_coef(
            step=train_step,
            start=getattr(args, "entropy_coef_start", 1e-2),
            end=getattr(args, "entropy_coef_end",   5e-4),
            anneal_after=getattr(args, "entropy_anneal_after", 5000),
            anneal_steps=getattr(args, "entropy_anneal_steps", 40000),
            schedule=getattr(args, "entropy_schedule", "linear"),
        )

        for tr in trainers:
            # 你的 actor 类里已有 self.entropy_coef（actor loss 用它做熵正则）
            if hasattr(tr, "actor"):
                tr.actor.entropy_coef = float(cur_entropy)
            # 兼容 COMADDPG：它的 loss 读取 trainer.entropy_coef
            setattr(tr, "entropy_coef", float(cur_entropy))

        # 可选：写到 TensorBoard 里，方便确认退火曲线
        if train_writer is not None:
            with train_writer.as_default():
                tf.summary.scalar("Entropy/Coef", cur_entropy, step=train_step)

        # 智能体决策
        if args.algorithm.lower() in ['ma2c', 'mappo']:
            # MA2C 的 action(obs, global_obs, mask)
            action_n = [tr.action(o, global_obs, m) for tr, o, m in zip(trainers, obs_n, mask_n)]
        elif args.algorithm.lower() == 'maca2c':
            # MACA2C：action(obs, mask) -> (actions_per_head, onehot_agent, pi_agent)
            # 上下文必须与当前观测 s_t 对齐，因此在 env.step() 前读取。
            maca2c_step_contexts = env.obs_encoder.get_coordination_contexts()
            if len(maca2c_step_contexts) != n_agents:
                raise RuntimeError(
                    f"Expected {n_agents} coordination contexts, "
                    f"got {len(maca2c_step_contexts)}."
                )
            action_n = []
            maca2c_step_agent_probs = []  # 这一环境步每个 agent 的 π_i
            for i, tr in enumerate(trainers):
                acts_i, onehot_i, pi_i = tr.action(obs_n[i], mask_n[i])
                action_n.append(acts_i)             # 环境吃的是“多头离散动作 id 列表”
                maca2c_step_agent_probs.append(pi_i)  # π_i 用于联合基线
            # 暂存当前步的策略分布，后面 env.step 之后会和 executed_action 一起拼联合向量
        else:
            action_n = [tr.action(o, m) for tr, o, m in zip(trainers, obs_n, mask_n)]
            last_actions = action_n.copy()  # 保存用于显示

        print(f"\n🎯 Episode {len(episode_rewards)-1}, Step {episode_step}")
        # print(f"   Generated actions: {action_n}")


        # (new_obs_n, new_global_obs), rew_n, done, _ = env.step(action_n)
        (new_obs_n, new_global_obs), rew_n, done, info = env.step(action_n)

        # ★ 若 env/模拟器提供了按智能体分解的奖励，则优先使用它
        if isinstance(info, dict) and ('agent_step_rewards' in info) and (info['agent_step_rewards'] is not None):
            rew_n = info['agent_step_rewards']

        # 优先用 info 里的被执行动作；若上层没透传，则回退到 temp_db 缓存；再不行才退回原动作
        executed_action_n = None
        if isinstance(info, dict):
            executed_action_n = info.get('executed_actions_multihead', None)
        if executed_action_n is None:
            executed_action_n = getattr(env.temp_db, 'last_executed_actions_multihead', None)
        if executed_action_n is None:
            # 最后的兜底（不建议长期使用）
            executed_action_n = action_n

        last_actions = executed_action_n.copy()  # 你后面有可视化，这里也同步成“被执行动作”

        # 便于人看：把 list[list[int]] 压成 tuple
        def _summ(a): 
            try: return [tuple(x) for x in a]
            except: return a

        # print(f"   Proposed actions: {_summ(action_n)}")
        # print(f"   ▶︎ Executed actions (used for replay): {_summ(executed_action_n)}")

        # # 差异统计（每个 agent 的多头是否一致）
        # try:
        #     diffs = [int(tuple(pa) != tuple(ea)) for pa, ea in zip(action_n, executed_action_n)]
        #     print(f"   Diff agents (proposed≠executed): {sum(diffs)}/{len(diffs)}")
        # except Exception as e:
        #     print(f"   [Warn] diff check failed: {e}")

        last_rewards = rew_n.copy()  # 保存用于显示

        # === 新增：计算当前步的全局 team reward（所有智能体 reward 之和） ===
        team_reward = float(np.sum(rew_n))
        # # 为了避免日志太多，这里只在前几个 episode 打印
        # if args.algorithm.lower() == "maca2c" and len(episode_rewards) <= 3:
        #     print(f"   per-agent rewards = {np.round(rew_n, 3)}")
        #     print(f"   [MACA2C][Debug] team_reward (sum of per-agent rewards) = {team_reward:.3f}")
        # input()

        # ===== 仅 MACA2C：把“已执行动作”变成 one-hot，并与策略分布拼成联合向量 =====
        if args.algorithm.lower() == "maca2c":
            # 每个 agent 的 multi-head one-hot（基于“真正执行”的动作）
            step_agent_onehots = [
                _multihead_onehot_from_ids(executed_action_n[i], head_dims_per_agent[i])
                for i in range(n_agents)
            ]
            # 联合 one-hot：所有 agent 的 one-hot 拼接
            joint_onehot = np.concatenate(step_agent_onehots, axis=-1).astype(np.float32)
            # 联合策略分布：用决策阶段缓存的每个 π_i 拼接
            joint_policy = np.concatenate(maca2c_step_agent_probs, axis=-1).astype(np.float32)

            maca2c_joint_onehots.append(joint_onehot)
            maca2c_joint_policies.append(joint_policy)
            maca2c_all_onehots.append(step_agent_onehots)
            maca2c_all_probs.append(maca2c_step_agent_probs)

        # === 把当前 obs_n / next_obs_n 补齐到统一长度（供部分算法使用） ===
        obs_all_pad, _ = pad1d_list(obs_n)          # (n_agents, obs_dim_pad)
        next_obs_all_pad, _ = pad1d_list(new_obs_n) # (n_agents, obs_dim_pad_next)
        next_mask_n = [env.get_mask(agent_index=i) for i in range(n_agents)] if args.use_iam else [None]*n_agents
            
        # 👉 新增：当前时刻 obs 也 pad 一份（comaddpg 需要联合 obs_n）
        obs_all_pad, _ = pad1d_list(obs_n)

        if args.algorithm.lower() == "comaddpg":
            # ✅ 只在 COMADDPG 下：每步只写一次“联合转移”
            trainers[0].experience(
                obs_all_pad,                    # obs_n: (n_agents, obs_dim_pad)
                global_obs,                     # global_obs: (G,)
                executed_action_n,              # act_n: list/array, 每个 agent 的多头离散动作 id 列表
                np.asarray(rew_n, np.float32),  # rew: (n_agents,) —— comaddpg 里会在 critic 端合成联合奖励
                next_obs_all_pad,               # new_obs_n: (n_agents, obs_dim_pad)
                new_global_obs,                 # new_global_obs: (G,)
                done,                           # done: bool
                mask_n,                         # mask_n: list[(total_nodes,)], head-0 的 IAM 掩码
                next_mask_n                     # new_mask_n
            )
        elif args.algorithm.lower() == "maca2c":
            # ✅ MACA2C：multi-agent Transformer critic 使用多 token 全局观测
            # 使用当前时刻所有智能体局部观测 obs_n 作为 token，补齐后得到 obs_all_pad
            # 形状为 (1, n_agents, token_dim)，与 MACA2CPolicy._encode_state 兼容
            maca2c_global_tokens = obs_all_pad[np.newaxis, :, :].astype(np.float32)
            for i, tr in enumerate(trainers):
                # if len(episode_rewards) <= 3:
                #     print(f"   [MACA2C][Debug] Agent {i} stores reward = {team_reward:.3f} (team_reward)")
                #     input()
                tr.experience(
                    obs=obs_n[i],
                    global_obs=maca2c_global_tokens,           # multi-token 形式的全局观测
                    actions_per_head=executed_action_n[i],     # list[int] per head
                    # reward=rew_n[i],
                    reward=team_reward,                        # ✅ 每个 agent 都用同一条全局 reward
                    done=done,
                    mask=mask_n[i],
                    coordination_context=maca2c_step_contexts[i]
                )
        else:
            if args.algorithm.lower() in ["ma2c", "mappo"]:
                # ✅ MA2C / MAPPO：experience 接口里没有 mask_all / next_mask_all
                for i, tr in enumerate(trainers):
                    tr.experience(
                        obs=obs_n[i],
                        global_obs=global_obs,
                        act_all=executed_action_n,
                        rew=rew_n[i],
                        next_obs=new_obs_n[i],
                        next_obs_all=next_obs_all_pad,
                        next_global_obs=new_global_obs,
                        done=done,
                    )
            else:
                # ✅ MADDPG 等 off-policy：继续使用带 mask 的接口
                for i, tr in enumerate(trainers):
                    tr.experience(
                        obs=obs_n[i],
                        global_obs=global_obs,
                        act_all=executed_action_n,
                        rew=rew_n[i],
                        next_obs=new_obs_n[i],
                        next_obs_all=next_obs_all_pad,
                        next_global_obs=new_global_obs,
                        done=done,
                        mask_all=mask_n,
                        next_mask_all=next_mask_n,
                    )
            # # 其它算法保持原来的“逐 agent 写入”不变
            # for i, tr in enumerate(trainers):
            #     tr.experience(
            #         obs=obs_n[i],                       # 本智能体当前观测
            #         global_obs=global_obs,              # 全局观测
            #         act_all=executed_action_n,          # 所有智能体本步动作（离散多头）
            #         rew=rew_n[i],                       # 单体回报
            #         next_obs=new_obs_n[i],              # 本智能体下一观测
            #         next_obs_all=next_obs_all_pad,      # （若你的旧接口需要）
            #         next_global_obs=new_global_obs,     # 全局下一观测
            #         done=done,
            #         mask_all=mask_n,
            #         next_mask_all=next_mask_n
            #     )

        obs_n = new_obs_n
        global_obs = new_global_obs
        episode_step += 1
        train_step += 1       

        # episode_rewards[-1] += float(np.sum(rew_n))  # 各智能体奖励求和=系统奖励
        episode_rewards[-1] += team_reward  # 各智能体奖励求和=系统奖励
        
        # 可视化当前步骤 - 根据参数和间隔决定
        if args.visualize and len(episode_rewards) % args.visualize_interval == 0:
            env.visualizer.visualize_step(
                episode=len(episode_rewards)-1,
                step=episode_step,
                slow_down_pls=(episode_step < 10),
                last_actions=last_actions,
                last_rewards=last_rewards
            )

        # # 每5步暂停让用户观察
        # if episode_step % 5 == 0 and episode_step > 0:
        #     print(f"\n📊 Episode {len(episode_rewards)-1}, Step {episode_step} Summary:")
        #     print(f"   Total reward so far: {episode_rewards[-1]:.3f}")
        #     print(f"   Step rewards: {rew_n}")
        #     print(f"   Actions taken: {last_actions}")
        #     print(f"   Vehicle positions: {env.temp_db.status_dict.get('v_coord', 'N/A')}")
        #     print(f"   Vehicle status ET: {env.temp_db.status_dict.get('ET', 'N/A')}")
        #     print(f"   Vehicle status ED: {env.temp_db.status_dict.get('ED', 'N/A')}")

        #     # 检查是否有节点被访问
        #     visited = getattr(env.temp_db, 'visited_nodes', set())
        #     delta = env.temp_db.get_val('delta') if hasattr(env.temp_db, 'get_val') else None
        #     print(f"   Visited nodes: {visited}")
        #     print(f"   Delta status: {delta}")

            # if episode_step < 2000:  # 前20步手动控制
            #     input("   Press Enter to continue...")
        
        # 训练智能体 - 修改这部分
        # ===== 参数更新（预热期内跳过）=====
        if args.algorithm.lower() in ["ma2c", "mappo"]:
            # On-policy：每步都尝试更新，update() 内部会在回合结束时才真正优化
            for tr in trainers:
                losses = tr.update(trainers, train_step)
                if losses is not None and len(losses) == 2:
                    critic_loss, actor_loss = losses
                    with train_writer.as_default():
                        if args.algorithm.lower() == "comaddpg":
                            # 只记录一条共享 critic 曲线（0 号返回的那条），其他 agent 不重复写
                            if tr.agent_index == 0:
                                tf.summary.scalar('loss/critic/shared', float(critic_loss), step=train_step)
                            # 每个智能体都写自己的 actor loss
                            tf.summary.scalar(f'loss/actor/agent_{tr.agent_index}',  float(actor_loss),  step=train_step)
                            tf.summary.scalar(f'loss/actor_abs/agent_{tr.agent_index}', abs(float(actor_loss)), step=train_step)
                        else:
                            tf.summary.scalar(f'Agent_{tr.agent_index}/Critic_Loss', float(critic_loss), step=train_step)
                            tf.summary.scalar(f'Agent_{tr.agent_index}/Actor_Loss', float(actor_loss), step=train_step)
                            tf.summary.scalar(f'Agent_{tr.agent_index}/Abs_Actor_Loss', abs(float(actor_loss)), step=train_step)
            train_writer.flush()
        elif args.algorithm.lower() == "maca2c":
            # MACA2C：on-policy，按照 rollout_len 或 回合结束进行分段更新
            rollout_len = getattr(args, "rollout_len", None)

            # 若配置了 rollout_len（>0），则当 buffer 中步数达到该长度时触发一次更新
            if rollout_len is not None and rollout_len > 0:
                buf_len = len(trainers[0].buffer)   # 所有 agent 的 buffer 长度相同

                if buf_len >= rollout_len:
                    # 此时 buffer 中存的正是上一段 rollout 的所有步，
                    # maca2c_joint_* 四个缓存也对应这段轨迹
                    seg_joint_onehots  = maca2c_joint_onehots
                    seg_joint_policies = maca2c_joint_policies
                    seg_all_onehots    = maca2c_all_onehots
                    seg_all_probs      = maca2c_all_probs

                    for tr in trainers:
                        actor_loss, critic_loss = tr.update(
                            joint_action_onehots=seg_joint_onehots,
                            joint_policies=seg_joint_policies,
                            all_agents_onehots=seg_all_onehots,
                            all_agents_probs=seg_all_probs
                        )
                        if actor_loss is not None and critic_loss is not None:
                            with train_writer.as_default():
                                tf.summary.scalar(
                                    f'Agent_{tr.agent_index}/Critic_Loss',
                                    float(critic_loss), step=train_step
                                )
                                tf.summary.scalar(
                                    f'Agent_{tr.agent_index}/Actor_Loss',
                                    float(actor_loss), step=train_step
                                )
                                tf.summary.scalar(
                                    f'Agent_{tr.agent_index}/Abs_Actor_Loss',
                                    abs(float(actor_loss)), step=train_step
                                )
                                tf.summary.scalar(
                                    f'Agent_{tr.agent_index}/Psi_Individual',
                                    tr.last_psi_ind, step=train_step
                                )
                                tf.summary.scalar(
                                    f'Agent_{tr.agent_index}/Psi_Joint',
                                    tr.last_psi_joint, step=train_step
                                )
                    train_writer.flush()

                    # 本段 rollout 已经用完：
                    # - 每个 trainer.update() 内部会 reset 自己的 buffer
                    # - 这里把多智能体联合信息的缓存也清空，准备下一段 rollout
                    maca2c_joint_onehots  = []
                    maca2c_joint_policies = []
                    maca2c_all_onehots    = []
                    maca2c_all_probs      = []
            # 如果 rollout_len 未设置或 buffer 尚不足，则当前步先不更新，
            # 留给后续步数累积或回合结束时统一处理
        else:
            def _get_bufsize(tr):
                f = getattr(tr, "buffer_size", None)
                if callable(f):
                    return int(f())
                # 兼容 MADDPG
                if hasattr(tr, "buffer"):
                    return len(tr.buffer)
                return 0

            if (_get_bufsize(trainers[0]) >= args.min_buffer_size and
                train_step % args.update_freq == 0 and
                train_step >= (args.warmup_steps or 0)):

            # # Off-policy（如 MADDPG）保持原来的最小缓冲/更新频率判断
            # if (len(trainers[0].buffer) >= args.min_buffer_size and
            #     train_step % args.update_freq == 0 and
            #     train_step >= (args.warmup_steps or 0)):  # ← 预热未结束时不更新

                for tr in trainers: 
                    tr.preupdate()
                
                # ⭐ 多次梯度步：默认 1 次；配置为 N 就做 N 次
                updates = int(args.updates_per_call or 1)
                for _ in range(updates):
                    for tr in trainers:
                        losses = tr.update(trainers, train_step, target_update_freq=args.target_update_freq)
                        if losses is None or len(losses) != 2:
                            # 本次没更新（例如缓冲不够或loss为NaN被跳过），直接下一个
                            continue

                        critic_loss, actor_loss = losses
                        with train_writer.as_default():
                            tf.summary.scalar(f'Agent_{tr.agent_index}/Critic_Loss', float(critic_loss), step=train_step)
                            tf.summary.scalar(f'Agent_{tr.agent_index}/Actor_Loss',  float(actor_loss),  step=train_step)
                            tf.summary.scalar(f'Agent_{tr.agent_index}/Abs_Actor_Loss', abs(float(actor_loss)), step=train_step)

                        print(f"   📊 Agent {tr.agent_index} - Critic Loss: {critic_loss:.6f}, Actor Loss: {actor_loss:.6f}")

                    train_writer.flush()
                     
                
            else:
                print(f"   Buffer size {len(trainers[0].buffer)} < min required {args.min_buffer_size}, skipping training")


        if done:
            print(f"\n🏁 Episode {len(episode_rewards)-1} completed!")
            print(f"   Final reward: {episode_rewards[-1]:.2f}")
            print(f"   Episode length: {episode_step}")
            termination_reason = info.get('termination_reason') if isinstance(info, dict) else None
            if termination_reason == 'max_steps':
                reason_text = 'Max steps reached'
            elif termination_reason == 'horizon':
                reason_text = 'Time horizon reached'
            elif termination_reason == 'deadlock_no_time_progress':
                reason_text = 'Deadlock: no time/state progress'
            elif termination_reason in {'action_decode_error', 'simulation_step_error'}:
                reason_text = f'Error: {termination_reason}'
            else:
                reason_text = 'Task completed'
            print(f"   Reason: {reason_text}")

            # ===== 仅 MACA2C：episode 结束时，对剩余未满 rollout_len 的尾巴做一次收尾更新 =====
            if args.algorithm.lower() == "maca2c":
                # 可能在 episode 中间已经做过多次 rollout 更新，
                # 这里如果 buffer 里还有数据，说明还有一小段尾巴没训练
                if len(trainers[0].buffer) > 0 and len(maca2c_joint_onehots) > 0:
                    for tr in trainers:
                        actor_loss, critic_loss = tr.update(
                            joint_action_onehots=maca2c_joint_onehots,
                            joint_policies=maca2c_joint_policies,
                            all_agents_onehots=maca2c_all_onehots,
                            all_agents_probs=maca2c_all_probs
                        )
                        if actor_loss is not None and critic_loss is not None:
                            with train_writer.as_default():
                                tf.summary.scalar(
                                    f'Agent_{tr.agent_index}/Critic_Loss',
                                    float(critic_loss), step=train_step
                                )
                                tf.summary.scalar(
                                    f'Agent_{tr.agent_index}/Actor_Loss',
                                    float(actor_loss), step=train_step
                                )
                                tf.summary.scalar(
                                    f'Agent_{tr.agent_index}/Abs_Actor_Loss',
                                    abs(float(actor_loss)), step=train_step
                                )
                                tf.summary.scalar(
                                    f'Agent_{tr.agent_index}/Psi_Individual',
                                    tr.last_psi_ind, step=train_step
                                )
                                tf.summary.scalar(
                                    f'Agent_{tr.agent_index}/Psi_Joint',
                                    tr.last_psi_joint, step=train_step
                                )
                    train_writer.flush()

                # 无论是否有尾巴，都清空联合缓存，下一集重新开始累计
                maca2c_joint_onehots  = []
                maca2c_joint_policies = []
                maca2c_all_onehots    = []
                maca2c_all_probs      = []
            
            # 获取成本统计
            cost_stats = env.reward_calc.get_episode_statistics()
            print(f"\n📊 Episode Cost Statistics:")
            print(f"   Total Cost (TC): {cost_stats['total_cost']:.2f}")
            print(f"   Travel Cost (Tra-C): {cost_stats['travel_cost']:.2f}")
            print(f"   Delay Penalty: {cost_stats['delay_penalty']:.2f}")
            print(f"   Unserved Penalty: {cost_stats['unserved_penalty']:.2f}")
            print(f"   Unserved Nodes: {cost_stats['unserved_count']}")
            print(f"   Served Nodes: {cost_stats['served_count']}")
            print(f"   Truck Travel Time: {cost_stats['truck_travel_time']:.2f}")
            print(f"   Drone Travel Time: {cost_stats['drone_travel_time']:.2f}")



            # 记录episode统计到TensorBoard
            episode_reward = episode_rewards[-1]

            # 基于原始总成本的“无权重 reward”：纯粹等于 -total_cost
            cost_based_reward = -cost_stats['total_cost']
            # 新增：真实 reward = 服务收益 − 原始成本（不带权重）
            real_reward = cost_stats.get('real_reward', None)

            episode_num = len(episode_rewards)-1
            if args.algorithm.lower() == "maca2c" and episode_num % 10 == 0:
                # Keep a low-frequency RSS baseline outside CMA-ES as well.
                gc.collect()
                _log_process_rss(f"main episode {episode_num}")
            
            with episode_writer.as_default():
                # 记录episode总奖励
                # 记录episode总奖励（带权重的：服务收益 − 加权成本）
                tf.summary.scalar('Episode/Total_Reward', episode_reward, step=episode_num)
                tf.summary.scalar('Episode/Episode_Length', episode_step, step=episode_num)
                # 新增：不加权的真实 reward 曲线
                if real_reward is not None:
                    tf.summary.scalar('Episode/Real_Reward', real_reward, step=episode_num)
                
                # 记录成本统计
                tf.summary.scalar('Costs/Total_Cost', cost_stats['total_cost'], step=episode_num)  
                tf.summary.scalar('Costs/Travel_Cost', cost_stats['travel_cost'], step=episode_num)
                tf.summary.scalar('Costs/Delay_Penalty', cost_stats['delay_penalty'], step=episode_num)
                tf.summary.scalar('Costs/Unserved_Penalty', cost_stats['unserved_penalty'], step=episode_num)
                tf.summary.scalar('Costs/Unserved_Count', cost_stats['unserved_count'], step=episode_num)
                tf.summary.scalar('Costs/Served_Count', cost_stats['served_count'], step=episode_num)
                tf.summary.scalar('Time/Truck_Travel_Time', cost_stats['truck_travel_time'], step=episode_num)
                tf.summary.scalar('Time/Drone_Travel_Time', cost_stats['drone_travel_time'], step=episode_num)
                
                # 记录环境统计
                try:
                    visited_nodes = len(getattr(env.temp_db, 'visited_nodes', set()))
                    delta = env.temp_db.get_val('delta')
                    total_nodes = len(delta)
                    completion_rate = visited_nodes / max(total_nodes - 1, 1)  # 排除depot
                    
                    tf.summary.scalar('Episode/Completion_Rate', completion_rate, step=episode_num)
                    tf.summary.scalar('Episode/Visited_Nodes', visited_nodes, step=episode_num)
                    # tf.summary.scalar('Episode/Remaining_Nodes', total_nodes - visited_nodes - 1, step=episode_num)
                    remaining = int(np.sum(env.temp_db.get_val('delta')[1:] == 1))
                    tf.summary.scalar('Episode/Remaining_Nodes', remaining, step=episode_num)
                except Exception as e:
                    print(f"   ⚠️  Could not record environment stats: {e}")
            
            # 强制写入TensorBoard
            train_writer.flush()
            episode_writer.flush()

            # 显示最终状态 - 根据参数决定
            if args.visualize and len(episode_rewards) % args.visualize_interval == 0:
                env.visualizer.visualize_step(
                    episode=len(episode_rewards)-1,
                    step=episode_step,
                    slow_down_pls=True,
                    last_actions=last_actions,
                    last_rewards=last_rewards
                )

            # CMA-ES 只在 episode 边界运行。候选产生的 actor/critic 更新
            # 会在比较结束后恢复，主训练仅部署本代最优 omega。
            completed_episodes = len(episode_rewards)
            if (args.algorithm.lower() == "maca2c" and
                    args.maca2c_cma_enabled and
                    completed_episodes % int(args.maca2c_cma_interval) == 0):
                print(
                    f"\n[CMA-ES] optimizing coordination weights after "
                    f"episode {completed_episodes}"
                )
                cma_result = _optimize_maca2c_weights(
                    env=env,
                    trainers=trainers,
                    args=args,
                    head_dims_per_agent=head_dims_per_agent,
                    weight_generator=coordination_weight_generator,
                    cma_optimizer=cma_weight_optimizer,
                    train_step=train_step,
                    completed_episodes=completed_episodes,
                )
                with episode_writer.as_default():
                    tf.summary.scalar(
                        'CMAES/Reward_Before', cma_result['reward_before'],
                        step=completed_episodes
                    )
                    tf.summary.scalar(
                        'CMAES/Best_Reward_Gain', cma_result['best_gain'],
                        step=completed_episodes
                    )
                    tf.summary.scalar(
                        'CMAES/Mean_Reward_Gain', cma_result['mean_gain'],
                        step=completed_episodes
                    )
                episode_writer.flush()
                print(
                    f"[CMA-ES] generation {cma_result['generation']} deployed: "
                    f"best gain={cma_result['best_gain']:.4f}"
                )
            

            # input("   Episode finished. Press Enter to start next episode...")
            obs_n, global_obs = env.reset()
            print(f"Episode {len(episode_rewards)-1} reward: {episode_rewards[-1]:.2f}")

            # 重新初始化 MACA2C 的联合缓存
            if args.algorithm.lower() == "maca2c":
                maca2c_joint_onehots = []
                maca2c_joint_policies = []
                maca2c_all_onehots = []
                maca2c_all_probs = []

            # 重置episode统计
            episode_step = 0
            episode_rewards.append(0.0)
            total_critic_loss = 0.0
            total_actor_loss = 0.0
            num_updates = 0

            # 每400个episode保存一次模型
            if len(episode_rewards) % args.save_rate == 0 and len(episode_rewards) > 1:
                for i, agent in enumerate(trainers):
                    if args.algorithm.lower() == "ma2c":
                        # MA2C：一个 policy 同时包含 actor+critic
                        agent.policy.save_weights(f"{save_path}/agent_{i}_policy_ep{len(episode_rewards)}.h5")
                    elif args.algorithm.lower() == "mappo":
                        # MAPPO：保存整个 policy（含 actor heads + critic）
                        agent.policy.save_weights(f"{save_path}/agent_{i}_mappo_policy_ep{len(episode_rewards)}.weights.h5")
                    elif args.algorithm.lower() == "maca2c":
                        # MACA2C：每个 agent 保存自己的 Actor；共享 Critic 只保存一次
                        agent.policy.save_weights(
                            f"{save_path}/agent_{i}_maca2c_actor_ep{len(episode_rewards)}.weights.h5"
                        )
                        if i == 0 and hasattr(agent, "critic"):
                            agent.critic.save_weights(
                                f"{save_path}/maca2c_shared_critic_ep{len(episode_rewards)}.weights.h5"
                            )
                    elif args.algorithm.lower() == "comaddpg":
                        # 每个 agent 各自保存 actor/target-actor（注意：ActorModel 本身就是 keras.Model）
                        for i, agent in enumerate(trainers):
                            agent.actor.save_weights(
                                f"{save_path}/agent_{i}_actor_weights_ep{len(episode_rewards)}.h5"
                            )
                            agent.target_actor.save_weights(
                                f"{save_path}/agent_{i}_actor_target_weights_ep{len(episode_rewards)}.h5"
                            )
                        # 共享 critic 只保存一次
                        if hasattr(trainers[0], "save_shared_critic"):
                            trainers[0].save_shared_critic(save_dir=save_path, prefix=f"ep{len(episode_rewards)}")
                    else:
                        agent.actor.model.save_weights(f"{save_path}/agent_{i}_actor_weights_ep{len(episode_rewards)}.h5")
                        agent.critic.model.save_weights(f"{save_path}/agent_{i}_critic_weights_ep{len(episode_rewards)}.h5")
                        agent.actor.target_model.save_weights(f"{save_path}/agent_{i}_actor_target_weights_ep{len(episode_rewards)}.h5")
                        agent.critic.target_model.save_weights(f"{save_path}/agent_{i}_critic_target_weights_ep{len(episode_rewards)}.h5")
                if args.algorithm.lower() == "maca2c":
                    coordination_weight_generator.save_parameters(
                        f"{save_path}/maca2c_coordination_omega_ep{len(episode_rewards)}.npz"
                    )
                    if cma_weight_optimizer is not None:
                        cma_weight_optimizer.save(
                            f"{save_path}/maca2c_cmaes_ep{len(episode_rewards)}.pkl"
                        )
                print(f"Models saved at episode {len(episode_rewards)}")

            # 【修改】加上判断，只有开启可视化时才调用
            if args.visualize:
                # 显示新episode的初始状态
                env.visualizer.visualize_step(
                    episode=len(episode_rewards)-1,
                    step=episode_step,
                    slow_down_pls=True,
                    last_actions=None,
                    last_rewards=None
                )

        # 达到最大训练轮数，保存模型并退出
        if len(episode_rewards) > args.num_episodes:
            print(f"\n🎓 Training completed after {args.num_episodes} episodes!")
            for i, agent in enumerate(trainers):
                if args.algorithm.lower() == "ma2c":
                    # MA2C：只保存 policy
                    agent.policy.save_weights(f"{save_path}/agent_{i}_ma2c_policy_final.h5")
                elif args.algorithm.lower() == "mappo":
                    agent.policy.save_weights(f"{save_path}/agent_{i}_mappo_policy_final.weights.h5")
                elif args.algorithm.lower() == "maca2c":
                    agent.policy.save_weights(
                        f"{save_path}/agent_{i}_maca2c_actor_final.weights.h5"
                    )
                    if i == 0 and hasattr(agent, "critic"):
                        agent.critic.save_weights(
                            f"{save_path}/maca2c_shared_critic_final.weights.h5"
                        )
                elif args.algorithm.lower() == "comaddpg":
                    # 只保存每个体的 actor/target-actor；共享 critic 用 0 号的接口保存一次
                    agent.actor.save_weights(f"{save_path}/agent_{i}_actor_final_weights.h5")
                    agent.target_actor.save_weights(f"{save_path}/agent_{i}_actor_target_final_weights.h5")
                    if i == 0 and hasattr(agent, "save_shared_critic"):
                        agent.save_shared_critic(save_dir=save_path, prefix="final")
                else:
                    # MADDPG：保存 actor/critic 及其 target
                    agent.actor.model.save_weights(f"{save_path}/agent_{i}_actor_final_weights.h5")
                    agent.critic.model.save_weights(f"{save_path}/agent_{i}_critic_final_weights.h5")
                    agent.actor.target_model.save_weights(f"{save_path}/agent_{i}_actor_target_final_weights.h5")
                    agent.critic.target_model.save_weights(f"{save_path}/agent_{i}_critic_target_final_weights.h5")
            if args.algorithm.lower() == "maca2c":
                coordination_weight_generator.save_parameters(
                    f"{save_path}/maca2c_coordination_omega_final.npz"
                )
                if cma_weight_optimizer is not None:
                    cma_weight_optimizer.save(
                        f"{save_path}/maca2c_cmaes_final.pkl"
                    )
            print("Final model weights saved.")

            # 关闭TensorBoard writers
            train_writer.close()
            episode_writer.close()
            print(f"📊 TensorBoard logs saved to: {log_dir}")
            break

if __name__ == '__main__':
    args = parse_args()
    train(args)
