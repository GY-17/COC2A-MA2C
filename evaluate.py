# evaluate.py
import argparse
import os
# 设置pygame使用虚拟显示驱动，避免在无头服务器上出错
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import json
from typing import Dict, List, Tuple
import seaborn as sns

# 导入您的环境和模型相关模块
from trucks_and_drones.build_env import BuildEnvironment
from trucks_and_drones.config import cfg
from maddpg.trainer.model import ActorModel, CriticModel
import copy
from gym import spaces

class ModelEvaluator:
    """用于评估训练好的MADDPG模型"""
    
    def __init__(self, checkpoint_dir, num_trucks, num_drones, total_nodes):
        self.checkpoint_dir = checkpoint_dir
        self.num_trucks = num_trucks
        self.num_drones = num_drones
        self.total_nodes = total_nodes
        self.n_agents = num_trucks + num_drones
        
        # 先创建一个临时环境来获取观测空间维度
        self.obs_dims = self._get_observation_dims()
        
        # 存储加载的模型
        self.actors = []
        self.critics = []
        
        # 加载模型
        self._load_models()
    
    def _get_observation_dims(self):
        """动态获取每个智能体的观测维度"""
        print("🔍 Auto-detecting observation dimensions...")
        
        # 创建临时环境
        env_cfg = cfg['environment']
        builder = (
            BuildEnvironment("VRPD_temp", grid=[10, 10])
            .trucks(num=env_cfg['num_trucks'])
            .drones(num=env_cfg['num_drones'])
            .depots(num=env_cfg.get('num_depots'))
            .customers(num=env_cfg.get('num_customers'))
            .observations()
            .actions()
            .rewards()
            .compile()
        )
        temp_env = builder.build()
        
        # 获取初始观测
        obs_n, global_obs = temp_env.reset()
        
        # 提取每个智能体的观测维度
        obs_dims = []
        for i, obs in enumerate(obs_n):
            obs_dim = obs.shape[0] if hasattr(obs, 'shape') else len(obs)
            obs_dims.append(obs_dim)
            
            agent_type = "Truck" if i < self.num_trucks else "Drone"
            print(f"  Agent {i} ({agent_type}): obs_dim = {obs_dim}")
        
        # 获取全局观测维度
        self.global_obs_dim = global_obs.shape[0] if hasattr(global_obs, 'shape') else len(global_obs)
        print(f"  Global observation dim: {self.global_obs_dim}")
        
        # 清理临时环境（如果有render窗口的话）
        if hasattr(temp_env, 'close'):
            temp_env.close()
        
        # 清理临时数据库连接等资源
        if hasattr(builder, 'temp_db'):
            builder.temp_db = None
        
        del temp_env
        del builder
        
        return obs_dims
        
    def _load_models(self):
        """加载保存的模型权重"""
        print(f"\n📂 Loading models from: {self.checkpoint_dir}")
        
        for i in range(self.n_agents):
            # 使用动态获取的观测维度
            obs_dim = self.obs_dims[i]
            
            # 确定智能体类型并创建正确的动作空间
            if i < self.num_trucks:
                # 卡车智能体
                act_space_list = [
                    spaces.Discrete(self.total_nodes),  # truck_target_node
                    spaces.Discrete(2)                   # truck_wait
                ]
                agent_type = "Truck"
                act_dim = self.total_nodes + 2
            else:
                # 无人机智能体
                act_space_list = [
                    spaces.Discrete(self.total_nodes),   # drone_service_node
                    spaces.Discrete(self.num_trucks),    # drone_rendezvous_truck
                    spaces.Discrete(2)                   # drone_continue
                ]
                agent_type = "Drone"
                act_dim = self.total_nodes + self.num_trucks + 2
            
            print(f"\n  Loading {agent_type} {i}:")
            print(f"    - Obs dim: {obs_dim} (auto-detected)")
            print(f"    - Act dim: {act_dim}")
            
            # 创建Actor模型
#             actor = ActorModel(obs_dim, act_space_list, 
#                              hidden_units=[1024, 512, 256], activation='relu')
            actor = ActorModel(obs_dim, act_space_list, 
                             hidden_units=[512, 256], activation='relu')
            
            # 构建模型（需要先调用一次来初始化权重）
            dummy_input = tf.zeros((1, obs_dim))
            _ = actor(dummy_input)

            # 加载Actor权重
            actor_path = os.path.join(self.checkpoint_dir, f'agent_{i}_actor_final_weights.h5')
            if os.path.exists(actor_path):
                try:
                    actor.load_weights(actor_path)
                    print(f"    ✅ Loaded actor weights")
                except Exception as e:
                    print(f"    ⚠️ Failed to load actor weights: {e}")
            else:
                # 尝试加载其他epoch的权重
                print(f"    ⚠️ Final weights not found, searching for latest checkpoint...")
                latest_actor_path = self._find_latest_checkpoint(i, 'actor')
                if latest_actor_path:
                    try:
                        actor.load_weights(latest_actor_path)
                        print(f"    ✅ Loaded actor weights from: {os.path.basename(latest_actor_path)}")
                    except Exception as e:
                        print(f"    ⚠️ Failed to load actor weights: {e}")
            
            self.actors.append(actor)
            
            # 计算Critic输入维度
            truck_act_dim = self.total_nodes + 2
            drone_act_dim = self.total_nodes + self.num_trucks + 2
            total_act_dim = truck_act_dim * self.num_trucks + drone_act_dim * self.num_drones
            critic_input_dim = self.global_obs_dim + total_act_dim
            
            # 创建和加载Critic模型
#             critic = CriticModel(critic_input_dim, hidden_units=[2048, 1024, 512, 256], activation='relu')
            critic = CriticModel(critic_input_dim, hidden_units=[1024, 512, 256], activation='relu')
            
            # 构建Critic模型
            dummy_critic_input = tf.zeros((1, critic_input_dim))
            _ = critic(dummy_critic_input)
            
            critic_path = os.path.join(self.checkpoint_dir, f'agent_{i}_critic_final_weights.h5')
            if os.path.exists(critic_path):
                try:
                    critic.load_weights(critic_path)
                    print(f"    ✅ Loaded critic weights")
                except:
                    pass
            else:
                latest_critic_path = self._find_latest_checkpoint(i, 'critic')
                if latest_critic_path:
                    try:
                        critic.load_weights(latest_critic_path)
                        print(f"    ✅ Loaded critic weights from: {os.path.basename(latest_critic_path)}")
                    except:
                        pass
            
            self.critics.append(critic)
    
    def _find_latest_checkpoint(self, agent_idx, model_type='actor'):
        """查找最新的检查点文件"""
        import glob
        import re
        
        # 搜索所有相关的权重文件
        pattern = os.path.join(self.checkpoint_dir, f'agent_{agent_idx}_{model_type}_weights_ep*.h5')
        files = glob.glob(pattern)
        
        if not files:
            pattern = os.path.join(self.checkpoint_dir, f'agent_{agent_idx}_{model_type}_*.h5')
            files = glob.glob(pattern)
        
        if not files:
            return None
        
        # 提取episode号并排序
        episodes = []
        for f in files:
            match = re.search(r'ep(\d+)', f)
            if match:
                episodes.append((int(match.group(1)), f))
        
        if episodes:
            # 返回最新的episode文件
            episodes.sort(key=lambda x: x[0], reverse=True)
            return episodes[0][1]
        
        # 如果没有找到episode号，返回最后修改的文件
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
    
    def select_actions(self, obs_n, env=None, deterministic=True, use_mask=True):
        """
        为所有智能体选择动作
        
        Args:
            obs_n: 所有智能体的观测
            env: 环境实例，用于获取mask
            deterministic: 是否使用确定性策略
            use_mask: 是否应用动作掩码
        """
        actions = []
        
        for i, (obs, actor) in enumerate(zip(obs_n, self.actors)):
            # 确保观测是正确的形状
            if obs.ndim == 1:
                obs = obs[np.newaxis, :]
            
            obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
            
            # 获取动作logits
            logits_heads = actor(obs_tensor)
            
            # 获取mask（如果启用且环境可用）
            mask = None
            if use_mask and env is not None:
                try:
                    mask = env.get_mask(agent_index=i)
                    print(f"    [Eval] Agent {i} mask: {np.where(mask)[0][:10]}...")  # 显示前10个有效索引
                except Exception as e:
                    print(f"    [Eval] Warning: Could not get mask for agent {i}: {e}")
                    mask = None
            
            # 对每个动作头选择动作
            agent_actions = []
            for head_idx, logits in enumerate(logits_heads):
                # 应用mask（仅对第一个动作头，即节点选择）
                if mask is not None and head_idx == 0:
                    # 确保mask长度匹配
                    if len(mask) != logits.shape[1]:
                        print(f"    [Eval] Warning: mask length {len(mask)} != logits dim {logits.shape[1]}")
                        # 调整mask大小
                        if len(mask) < logits.shape[1]:
                            extended_mask = np.zeros(logits.shape[1], dtype=bool)
                            extended_mask[:len(mask)] = mask
                            mask = extended_mask
                        else:
                            mask = mask[:logits.shape[1]]
                    
                    # 将无效动作的logits设为极小值
                    masked_logits = logits.numpy()[0].copy()
                    masked_logits[~mask] = -1e9
                    logits_tensor = tf.constant([masked_logits])
                else:
                    logits_tensor = logits
                
                if deterministic:
                    # 确定性策略：选择概率最高的动作
                    action = tf.argmax(logits_tensor, axis=-1).numpy()[0]
                else:
                    # 随机策略：根据概率分布采样
                    probs = tf.nn.softmax(logits_tensor)
                    action = tf.random.categorical(tf.math.log(probs + 1e-8), 1).numpy()[0, 0]
                
                # 对第一个动作头进行额外的有效性检查
                if mask is not None and head_idx == 0:
                    if action < len(mask) and not mask[action]:
                        # 如果选择了无效动作，回退到有效动作
                        valid_indices = np.where(mask)[0]
                        if len(valid_indices) > 0:
                            action = np.random.choice(valid_indices) if not deterministic else valid_indices[0]
                            print(f"    [Eval] Agent {i} head {head_idx}: fallback to action {action}")
                
                agent_actions.append(int(action))
            
            actions.append(agent_actions)
        
        return actions

class SensitivityAnalyzer:
    """敏感性分析类"""
    
    def __init__(self, evaluator, base_config):
        self.evaluator = evaluator
        self.base_config = copy.deepcopy(base_config)
        self.results = {}
        

    def create_env(self, custom_config=None):
        """创建环境实例"""
        import copy
        
        # 深拷贝基础配置，避免修改全局配置
        config = copy.deepcopy(self.base_config)
        
        # 应用自定义配置
        if custom_config:
            # 特殊处理环境配置
            if 'environment' not in config:
                config['environment'] = {}
            
            for key, value in custom_config.items():
                if key in ['WD_max', 'max_charge', 'truck_speed', 'drone_speed', 'ct_cost', 'cd_cost']:
                    config['environment'][key] = value
                elif key == 'dynamic_nodes':
                    config['environment']['dynamic_nodes'] = value
                elif key == 'node_alpha':  # 处理alpha参数
                    # 更新所有customer节点的alpha值
                    if 'node' not in config:
                        config['node'] = {}
                    if 'customer' not in config['node']:
                        config['node']['customer'] = {}
                    config['node']['customer']['alpha'] = value
                else:
                    config['environment'][key] = value
        
        # 获取更新后的环境配置
        env_cfg = config['environment']
        # Evaluation uses the standard instance unless a caller supplies a
        # specific dataset path (for example, a batch evaluation).
        if not env_cfg.get('dataset_path'):
            env_cfg['dataset_path'] = env_cfg.get('evaluation_dataset_path')
        
        # ⭐️ 关键修改：在创建builder之前先更新全局cfg
        original_cfg = cfg.copy()
        cfg.clear()
        cfg.update(config)

        # ⭐️ 新增：特殊处理DoD相关的配置更新
        if custom_config and 'dynamic_nodes' in custom_config:
            dynamic_config = custom_config['dynamic_nodes']
            
            # 重新计算静态和动态客户节点数量
            total_customers = env_cfg.get('num_customers', 30)
            if 'dod' not in dynamic_config:
                raise ValueError(f"DoD value missing in dynamic_config: {dynamic_config}")
            new_dod = dynamic_config['dod']  # 直接获取，确保传入了正确的值
            
            print(f"📋 Processing DoD: {new_dod} (from config: {dynamic_config})")
            
            # 根据新的DoD重新分配节点
            num_static = int(total_customers * (1 - new_dod))
            num_dynamic = total_customers - num_static
            
            # 更新配置中的节点数量分配
            cfg['environment']['num_static_customers'] = num_static
            cfg['environment']['num_dynamic_customers'] = num_dynamic
            cfg['environment']['dynamic_nodes'] = dynamic_config
            
            print(f"🎯 DoD Configuration Updated:")
            print(f"   New DoD: {new_dod}")
            print(f"   Total customers: {total_customers}")
            print(f"   Static customers: {num_static}")
            print(f"   Dynamic customers: {num_dynamic}")
        
        
        try:
            # 现在创建builder，此时cfg已经是更新后的值
            builder = (
                BuildEnvironment("VRPD_Eval", grid=[10, 10])
                .trucks(num=env_cfg['num_trucks'], 
                        speed=env_cfg.get('truck_speed'))
                .drones(num=env_cfg['num_drones'], 
                        speed=env_cfg.get('drone_speed'),
                        max_cargo=env_cfg.get('WD_max'),      # 明确传递WD_max
                        max_charge=env_cfg.get('max_charge'))  # 明确传递max_charge
                .depots(num=env_cfg.get('num_depots'))
                .customers(num=env_cfg.get('num_customers'))
            
                .observations()
                .actions()
                .rewards()
                .compile()  # 现在compile时，temp_db会读取到正确的cfg值
            )
            
            env = builder.build()
            
        finally:
            # 确保恢复原始配置
            cfg.clear()
            cfg.update(original_cfg)
        
        return env
    
    def run_episode(self, env, max_steps=500, render=False):
        """运行单个episode"""
        obs_n, global_obs = env.reset()
        
        episode_metrics = {
            'total_reward': 0,
            'total_cost': 0,  # 新增：添加total_cost字段
            'travel_cost': 0,
            'delay_penalty': 0,
            'service_reward': 0,
            'unserved_count': 0,
            'served_count': 0,
            'truck_travel_time': 0,
            'drone_travel_time': 0,
            'episode_length': 0,
            'episode_length': 0,           # 步数
            'episode_total_time': 0.0      # ☆ 新增：episode推进的总时间（连续时间）
        }
        # 添加调试暂停
        print(f"\n🎬 Episode开始. 初始观测维度: {[o.shape for o in obs_n]}")
        # input("按Enter键开始episode评估...")
        
        for step in range(max_steps):
            # 选择动作 - 传递env以使用mask
            actions = self.evaluator.select_actions(obs_n, env=env, deterministic=True, use_mask=True)
        
            # # 添加每步调试信息
            # if step < 10:  # 前5步或每100步暂停
            #     print(f"\n📍 步骤 {step}/{max_steps}")
            #     print(f"   选择的动作: {actions}")
            #     print(f"   当前累积奖励: {episode_metrics['total_reward']:.3f}")
            #     input("   按Enter键执行动作...")
            # 执行动作
            (new_obs_n, new_global_obs), rewards, done, info = env.step(actions)
            
            # 更新指标
            # ☆ 累计所有智能体的奖励，而不是只用 rewards[0]
            try:
                episode_metrics['total_reward'] += float(np.sum(rewards))
            except Exception:
                # 某些环境 reward 不是可加的数组，就跳过
                pass
            episode_metrics['episode_length'] += 1
            
            # 获取成本/服务统计（每步更新一次最新值）
            try:
                if hasattr(env, 'reward_calc') and hasattr(env.reward_calc, 'get_episode_statistics'):
                    stats = env.reward_calc.get_episode_statistics()
                    episode_metrics['total_cost'] = stats.get('total_cost', 0)  # 新增：保存total_cost
                    episode_metrics['travel_cost'] = stats.get('travel_cost', 0)
                    episode_metrics['delay_penalty'] = stats.get('delay_penalty', 0)
                    episode_metrics['service_reward'] = stats.get('service_reward', 0)
                    episode_metrics['unserved_count'] = stats.get('unserved_count', 0)
                    episode_metrics['served_count'] = stats.get('served_count', 0)
                    episode_metrics['truck_travel_time'] = stats.get('truck_travel_time', 0)
                    episode_metrics['drone_travel_time'] = stats.get('drone_travel_time', 0)
                else:
                    # 从其他来源获取统计信息
                    if hasattr(env, 'temp_db'):
                        delta = env.temp_db.get_val('delta')
                        episode_metrics['served_count'] = sum(1 for d in delta if d == 0) - 1  # 排除depot
                        episode_metrics['unserved_count'] = sum(1 for d in delta if d == 1)
            except Exception as e:
                print(f"Warning: Could not get detailed statistics: {e}")
            
            if render:
                try:
                    env.render()
                except Exception as e:
                    print(f"Warning: Could not render - {e}")
                    render = False  # 禁用后续渲染尝试
            
            if done:
                print(f"\n🏁 Episode在步骤 {step} 完成")
                print(f"   最终奖励: {episode_metrics['total_reward']:.3f}")
                print(f"   服务/未服务节点: {episode_metrics['served_count']}/{episode_metrics['unserved_count']}")
                # input("   按Enter键结束episode...")
                # 记录本次 episode 推进的总时间（连续时间）
                if hasattr(env, 'temp_db') and hasattr(env.temp_db, 'total_time'):
                    episode_metrics['episode_total_time'] = float(env.temp_db.total_time)
                
                # ☆ 关键：终止后立刻跳出循环，不要再 step 一次
                break
            
             # 7) 没结束则推进观测
            obs_n = new_obs_n
        else:
            # for 循环自然结束（达到 max_steps）也记一下当前的连续时间
            if hasattr(env, 'temp_db') and hasattr(env.temp_db, 'total_time'):
                episode_metrics['episode_total_time'] = float(env.temp_db.total_time)

        
        return episode_metrics
    
    def run_evaluation(self, custom_config=None, num_episodes=20, render=False):
        """运行多个episode的评估"""
        print(f"\n🔄 使用配置创建环境: {custom_config}")
        # input("按Enter键创建环境...")
        env = self.create_env(custom_config)
        
        all_metrics = []
        for episode in range(num_episodes):
            print(f"\n{'='*60}")
            print(f"📊 Episode {episode + 1}/{num_episodes}")
            print(f"{'='*60}")
            
            # # 每5个episode暂停一次
            # if episode % 5 == 0:
            #     input(f"按Enter键开始episode {episode + 1}...")
            metrics = self.run_episode(env, render=False)
            all_metrics.append(metrics)
            # 显示当前episode结果
            print(f"Episode {episode + 1} 指标:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.2f}")
            
        # 计算平均值
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
        
        print(f"\n📈 评估完成. 平均指标:")
        for key, value in avg_metrics.items():
            if key.startswith('avg_'):
                print(f"  {key}: {value:.2f}")
        # input("按Enter键继续...")
        
        return avg_metrics

    def analyze_cost_ratio(self, cost_ratio_range=None):
        """分析无人机与卡车旅行成本比(cd/ct)的影响 - 保持总成本不变"""
        if cost_ratio_range is None:
            cost_ratio_range = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
        
        print("\n💰 Analyzing Travel Cost Ratio Impact (cd/ct)")
        print("-" * 50)
        
        # 🔑 关键修改：固定总成本
        base_truck_cost = self.base_config['environment']['ct_cost']
        base_drone_cost = self.base_config['environment']['cd_cost']
        TOTAL_COST = base_truck_cost + base_drone_cost  # 例如：1.0 + 0.5 = 1.5
        
        results = []
        for ratio in cost_ratio_range:
            print(f"\n  Testing cost ratio: {ratio}")
            
            # 🔑 保持 ct + cd = TOTAL_COST
            # cd/ct = ratio  =>  cd = ratio * ct
            # ct + ratio * ct = TOTAL_COST
            # ct * (1 + ratio) = TOTAL_COST
            ct = TOTAL_COST / (1 + ratio)
            cd = ratio * ct
            
            custom_config = {
                'ct_cost': ct,
                'cd_cost': cd
            }
            
            print(f"    ct_cost: {ct:.4f}, cd_cost: {cd:.4f}, sum: {ct+cd:.4f}")
            
            metrics = self.run_evaluation(custom_config, num_episodes=10)
            
            result = {
                'cost_ratio': ratio,
                'ct_cost': ct,
                'cd_cost': cd,
                'total_cost': metrics['avg_total_cost'],
                'travel_cost': metrics['avg_travel_cost'],
                'delay_penalty': metrics['avg_delay_penalty'],
                'served_nodes': metrics['avg_served_count'],
                'unserved_nodes': metrics['avg_unserved_count'],
                'service_rate': metrics['avg_served_count'] / 
                            (metrics['avg_served_count'] + metrics['avg_unserved_count'] + 1e-6)
            }
            results.append(result)
            
            print(f"    Total Cost: {result['total_cost']:.2f}")
            print(f"    Travel Cost: {result['travel_cost']:.2f}")
            print(f"    Service Rate: {result['service_rate']:.2%}")
        
        self.results['cost_ratio'] = pd.DataFrame(results)
        return self.results['cost_ratio']
    
#     def analyze_cost_ratio(self, cost_ratio_range=None):
#         """分析无人机与卡车旅行成本比(cd/ct)的影响"""
#         if cost_ratio_range is None:
#             cost_ratio_range = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
        
#         print("\n💰 Analyzing Travel Cost Ratio Impact (cd/ct)")
#         print("-" * 50)
        
#         base_truck_cost = self.base_config['environment']['ct_cost']
        
#         results = []
#         for ratio in cost_ratio_range:
#             print(f"\n  Testing cost ratio: {ratio}")
            
#             custom_config = {
#                 'ct_cost': base_truck_cost,  # 保持卡车成本不变
#                 'cd_cost': base_truck_cost * ratio  # 调整无人机成本
#             }
            
#             metrics = self.run_evaluation(custom_config, num_episodes=10)
            
#             result = {
#                 'cost_ratio': ratio,
#                 'total_cost': metrics['avg_total_cost'],
#                 'travel_cost': metrics['avg_travel_cost'],
#                 'delay_penalty': metrics['avg_delay_penalty'],
#                 'served_nodes': metrics['avg_served_count'],
#                 'unserved_nodes': metrics['avg_unserved_count'],
#                 'service_rate': metrics['avg_served_count'] / 
#                             (metrics['avg_served_count'] + metrics['avg_unserved_count'] + 1e-6)
#             }
#             results.append(result)
            
#             print(f"    Total Cost: {result['total_cost']:.2f}")
#             print(f"    Travel Cost: {result['travel_cost']:.2f}")
#             print(f"    Service Rate: {result['service_rate']:.2%}")
        
#         self.results['cost_ratio'] = pd.DataFrame(results)
#         return self.results['cost_ratio']

    def analyze_delay_penalty(self, alpha_range=None):
        """分析延迟惩罚系数(alpha)的影响"""
        if alpha_range is None:
            alpha_range = [0, 0.25, 0.5, 1.0, 1.5, 2]
        
        print("\n⏱️ Analyzing Delay Penalty Coefficient Impact")
        print("-" * 50)
        
        results = []
        for alpha in alpha_range:
            print(f"\n  Testing alpha: {alpha}")
            
            # 需要修改node配置中的alpha值
            custom_config = {
                'node_alpha': alpha  # 传递新的alpha值
            }
            
            metrics = self.run_evaluation(custom_config, num_episodes=10)
            
            # 计算平均延迟时间
            avg_delay_time = metrics.get('avg_delay_time', 0)
            if avg_delay_time == 0 and 'avg_delay_penalty' in metrics and alpha > 0:
                # 如果没有直接的延迟时间，可以从延迟惩罚反推
                avg_delay_time = metrics['avg_delay_penalty'] / alpha
            
            result = {
                'alpha': alpha,
                'total_cost': metrics['avg_total_cost'],
                'travel_cost': metrics['avg_travel_cost'],
                'delay_penalty': metrics['avg_delay_penalty'],
                'delay_time': avg_delay_time,
                'served_nodes': metrics['avg_served_count'],
                'unserved_nodes': metrics['avg_unserved_count'],
                'service_rate': metrics['avg_served_count'] / 
                            (metrics['avg_served_count'] + metrics['avg_unserved_count'] + 1e-6)
            }
            results.append(result)
            
            print(f"    Total Cost: {result['total_cost']:.2f}")
            print(f"    Delay Penalty: {result['delay_penalty']:.2f}")
            print(f"    Avg Delay Time: {result['delay_time']:.2f}")
            print(f"    Service Rate: {result['service_rate']:.2%}")
        
        self.results['delay_penalty'] = pd.DataFrame(results)
        return self.results['delay_penalty']
    
    def analyze_drone_capacity(self, capacity_range=None):
        """分析无人机载重容量的影响"""
        if capacity_range is None:
            capacity_range = [30, 40, 50, 60, 70, 80]
            # capacity_range = [30]
        
        print("\n🚁 Analyzing Drone Capacity Impact")
        print("-" * 50)
        # input("按Enter键开始无人机载重分析...")
        
        results = []
        for i, capacity in enumerate(capacity_range):
            print(f"\n📦 测试载重: {capacity} kg ({i+1}/{len(capacity_range)})")
            # input(f"按Enter键测试载重 {capacity}...")
            
            custom_config = {'WD_max': capacity}
            metrics = self.run_evaluation(custom_config, num_episodes=10)
            
            result = {
                'capacity': capacity,
                'total_cost': metrics['avg_total_cost'],  # 负奖励转为成本
                'travel_cost': metrics['avg_travel_cost'],
                'delay_penalty': metrics['avg_delay_penalty'],
                'served_nodes': metrics['avg_served_count'],
                'unserved_nodes': metrics['avg_unserved_count'],
                'service_rate': metrics['avg_served_count'] / 
                               (metrics['avg_served_count'] + metrics['avg_unserved_count'] + 1e-6)
            }
            results.append(result)
            
            print(f"    总成本: {result['total_cost']:.2f}")
            print(f"    服务率: {result['service_rate']:.2%}")
            # input(f"    按Enter键继续下一个载重测试...")
        
        self.results['drone_capacity'] = pd.DataFrame(results)
        print("\n✅ 无人机载重分析完成!")
        # input("按Enter键继续...")
        return self.results['drone_capacity']
    
    def analyze_battery_capacity(self, battery_range=None):
        """分析无人机电池容量的影响"""
        if battery_range is None:
            battery_range = [10,20, 30, 40, 50, 60, 70, 80]
            # battery_range = [20]
        
        print("\n🔋 Analyzing Battery Capacity Impact")
        print("-" * 50)
        
        results = []
        for battery in battery_range:
            print(f"\n  Testing battery: {battery} km")
            
            custom_config = {'max_charge': battery}
            metrics = self.run_evaluation(custom_config, num_episodes=10)
            
            result = {
                'battery': battery,
                'total_cost': metrics['avg_total_cost'],
                'travel_cost': metrics['avg_travel_cost'],
                'delay_penalty': metrics['avg_delay_penalty'],
                'served_nodes': metrics['avg_served_count'],
                'unserved_nodes': metrics['avg_unserved_count'],
                'service_rate': metrics['avg_served_count'] / 
                               (metrics['avg_served_count'] + metrics['avg_unserved_count'] + 1e-6)
            }
            results.append(result)
            
            print(f"    Total Cost: {result['total_cost']:.2f}")
            print(f"    Service Rate: {result['service_rate']:.2%}")
        
        self.results['battery_capacity'] = pd.DataFrame(results)
        return self.results['battery_capacity']
    
    def analyze_speed_ratio(self, ratio_range=None):
        """分析无人机与卡车速度比的影响"""
        if ratio_range is None:
            ratio_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            # ratio_range = [0.5]
        
        print("\n⚡ Analyzing Speed Ratio Impact")
        print("-" * 50)
        
        base_truck_speed = self.base_config['environment']['truck_speed']
        
        results = []
        for ratio in ratio_range:
            print(f"\n  Testing speed ratio: {ratio}")
            
            custom_config = {
                'truck_speed': base_truck_speed,
                'drone_speed': base_truck_speed * ratio
            }
            metrics = self.run_evaluation(custom_config, num_episodes=10)
            
            result = {
                'speed_ratio': ratio,
                'total_cost': metrics['avg_total_cost'],
                'travel_cost': metrics['avg_travel_cost'],
                'truck_time': metrics['avg_truck_travel_time'],
                'drone_time': metrics['avg_drone_travel_time'],
                'service_rate': metrics['avg_served_count'] / 
                               (metrics['avg_served_count'] + metrics['avg_unserved_count'] + 1e-6)
            }
            results.append(result)
            
            print(f"    Total Cost: {result['total_cost']:.2f}")
            print(f"    Service Rate: {result['service_rate']:.2%}")
        
        self.results['speed_ratio'] = pd.DataFrame(results)
        return self.results['speed_ratio']
    
    def analyze_damage_dod(self,
                        damage_ratios=None,
                        dod_values=None,
                        episodes=10):
        """
        固定训练用的节点规模 n（不改 num_customers），
        在不同 λ（road_damage_ratio）与 DoD（dynamic_nodes.dod）下评估：
        total_cost, travel_cost, delay_penalty, served_nodes, unserved_nodes,
        service_rate, episode_total_time（本 episode 推进总时间）
        """
        import pandas as pd

        if damage_ratios is None:
            damage_ratios = [0.1, 0.3, 0.5]
        if dod_values is None:
            dod_values = [0.2, 0.5, 0.7]

        # 从基础配置里读取训练用的 n（不去覆盖它）
        train_n = int(self.base_config['environment'].get('num_customers', 15))

        print("\n🧪 Analyze grid over [λ × DoD] (fixed n)")
        print(f"   fixed n={train_n}, λ={damage_ratios}, DoD={dod_values}")

        rows = []
        for lam in damage_ratios:
            for dod in dod_values:
                print(f"\n  ▶ Config: n={train_n} (fixed), λ={lam}, DoD={dod}")

                # 仅覆盖 λ 与 DoD，不改 num_customers
                custom = {
                    'road_damage_ratio': float(lam),
                    'dynamic_nodes': {
                        'enable': True,
                        'dod': float(dod),
                        # 复用原配置里的 delta_t
                        'delta_t': self.base_config['environment'].get('dynamic_nodes', {}).get('delta_t', 5),
                    },
                }

                m = self.run_evaluation(custom_config=custom,
                                        num_episodes=episodes,
                                        render=False)

                served = float(m.get('avg_served_count', 0.0))
                unserved = float(m.get('avg_unserved_count', 0.0))
                service_rate = served / (served + unserved + 1e-6)

                total_episode_time = float(m.get('avg_episode_total_time', 0.0))

                rows.append({
                    'n': train_n,                         # 记录固定的 n，便于表头展示
                    'λ': float(lam),
                    'DoD': float(dod),
                    'total_cost': float(m.get('avg_total_cost', 0.0)),
                    'travel_cost': float(m.get('avg_travel_cost', 0.0)),
                    'delay_penalty': float(m.get('avg_delay_penalty', 0.0)),
                    'served_nodes': served,
                    'unserved_nodes': unserved,
                    'service_rate': service_rate,
                    'total_episode_time': total_episode_time
                })

                print(f"    total_cost={m.get('avg_total_cost', 0.0):.2f}, "
                    f"travel_cost={m.get('avg_travel_cost', 0.0):.2f}, "
                    f"delay_penalty={m.get('avg_delay_penalty', 0.0):.2f}, "
                    f"service_rate={service_rate:.2%}, "
                    f"episode_total_time={total_episode_time:.2f}")

        df = pd.DataFrame(rows)
        self.results['damage_dod'] = df
        return df

    def analyze_dynamic_degree(self, dod_range=None):
        """分析动态度(DoD)的影响"""
        if dod_range is None:
            dod_range = [0.0, 0.2, 0.5, 0.7]
            # dod_range = [0.0]
        
        print("\n🎯 Analyzing Degree of Dynamism Impact")
        print("-" * 50)
        
        results = []
        for dod in dod_range:
            print(f"\n  Testing DoD: {dod}")
            
            # 正确的方式：只传递需要修改的配置项
            custom_config = {
                'dynamic_nodes': {
                    'enable': True,
                    'dod': dod,
                    'delta_t': 5
                }
            }
            
            # 直接传递custom_config，而不是整个env_config
            metrics = self.run_evaluation(custom_config, num_episodes=10)
            
            result = {
                'dod': dod,
                'total_cost': metrics['avg_total_cost'],
                'travel_cost': metrics['avg_travel_cost'],
                'delay_penalty': metrics['avg_delay_penalty'],
                'served_nodes': metrics['avg_served_count'],
                'unserved_nodes': metrics['avg_unserved_count'],
                'service_rate': metrics['avg_served_count'] / 
                               (metrics['avg_served_count'] + metrics['avg_unserved_count'] + 1e-6)
            }
            results.append(result)
            
            print(f"    Total Cost: {result['total_cost']:.2f}")
            print(f"    Service Rate: {result['service_rate']:.2%}")
        
        self.results['dynamic_degree'] = pd.DataFrame(results)
        return self.results['dynamic_degree']
    
    def plot_all_results(self, save_dir='evaluation_results'):
        """绘制所有分析结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 修复：使用兼容的样式设置
        try:
            # 尝试使用新版本的seaborn样式
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                # 尝试旧版本的seaborn样式
                plt.style.use('seaborn-darkgrid')
            except:
                # 如果都失败，使用默认样式
                plt.style.use('ggplot')
                print("注意：使用默认ggplot样式，因为seaborn样式不可用")
        
        # 设置颜色调色板
        try:
            import seaborn as sns
            sns.set_palette("husl")
        except ImportError:
            # 如果没有seaborn，使用matplotlib默认调色板
            pass
        # 创建子图
        fig = plt.figure(figsize=(18, 14))
        
        # 1. 无人机载重容量分析
        if 'drone_capacity' in self.results:
            ax1 = plt.subplot(2, 3, 1)
            df = self.results['drone_capacity']
            ax1.plot(df['capacity'], df['total_cost'], 'o-', linewidth=2, markersize=8, label='Total Cost')
            ax1.set_xlabel('Drone Capacity (kg)', fontsize=12)
            ax1.set_ylabel('Total Cost', fontsize=12)
            ax1.set_title('Impact of Drone Capacity', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # 添加服务率的双Y轴
            ax1_twin = ax1.twinx()
            ax1_twin.plot(df['capacity'], df['service_rate'], 's-', color='red', 
                         linewidth=2, markersize=6, alpha=0.7, label='Service Rate')
            ax1_twin.set_ylabel('Service Rate', fontsize=12, color='red')
            ax1_twin.tick_params(axis='y', labelcolor='red')
        
        # 2. 电池容量分析
        if 'battery_capacity' in self.results:
            ax2 = plt.subplot(2, 3, 2)
            df = self.results['battery_capacity']
            ax2.plot(df['battery'], df['total_cost'], 'o-', linewidth=2, markersize=8)
            ax2.set_xlabel('Battery Capacity (km)', fontsize=12)
            ax2.set_ylabel('Total Cost', fontsize=12)
            ax2.set_title('Impact of Battery Capacity', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 3. 速度比分析
        if 'speed_ratio' in self.results:
            ax3 = plt.subplot(2, 3, 3)
            df = self.results['speed_ratio']
            ax3.plot(df['speed_ratio'], df['total_cost'], 'o-', linewidth=2, markersize=8)
            ax3.set_xlabel('Speed Ratio (Drone/Truck)', fontsize=12)
            ax3.set_ylabel('Total Cost', fontsize=12)
            ax3.set_title('Impact of Speed Ratio', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. 动态度分析
        if 'dynamic_degree' in self.results:
            ax4 = plt.subplot(2, 3, 4)
            df = self.results['dynamic_degree']
            ax4.plot(df['dod'], df['total_cost'], 'o-', linewidth=2, markersize=8)
            ax4.set_xlabel('Degree of Dynamism', fontsize=12)
            ax4.set_ylabel('Total Cost', fontsize=12)
            ax4.set_title('Impact of Dynamic Nodes', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 5. 成本组成分析（堆叠条形图）
        if len(self.results) > 0:
            ax5 = plt.subplot(2, 3, 5)
            # 取第一个分析的数据作为示例
            first_key = list(self.results.keys())[0]
            df = self.results[first_key]
            
            if 'travel_cost' in df.columns and 'delay_penalty' in df.columns:
                x_pos = np.arange(len(df))
                width = 0.35
                
                ax5.bar(x_pos - width/2, df['travel_cost'], width, label='Travel Cost')
                ax5.bar(x_pos + width/2, df['delay_penalty'], width, label='Delay Penalty')
                
                ax5.set_xlabel('Configuration Index', fontsize=12)
                ax5.set_ylabel('Cost Components', fontsize=12)
                ax5.set_title('Cost Breakdown', fontsize=14, fontweight='bold')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        # 6. 服务率对比
        if len(self.results) > 0:
            ax6 = plt.subplot(2, 3, 6)
            for key, df in self.results.items():
                if 'service_rate' in df.columns:
                    label = key.replace('_', ' ').title()
                    ax6.plot(df.index, df['service_rate'], 'o-', linewidth=2, 
                            markersize=6, label=label)
            
            ax6.set_xlabel('Configuration Index', fontsize=12)
            ax6.set_ylabel('Service Rate', fontsize=12)
            ax6.set_title('Service Rate Comparison', fontsize=14, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. 旅行成本比分析
        if 'cost_ratio' in self.results:
            ax7 = plt.subplot(3, 3, 7)
            df = self.results['cost_ratio']
            ax7.plot(df['cost_ratio'], df['total_cost'], 'o-', linewidth=2, markersize=8, label='Total Cost')
            ax7.plot(df['cost_ratio'], df['travel_cost'], 's-', linewidth=2, markersize=6, label='Travel Cost')
            ax7.set_xlabel('Cost Ratio (cd/ct)', fontsize=12)
            ax7.set_ylabel('Cost', fontsize=12)
            ax7.set_title('Impact of Travel Cost Ratio', fontsize=14, fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            ax7.axvline(x=2.0, color='red', linestyle='--', alpha=0.5, label='Stabilization Point')

        # 8. 延迟惩罚系数分析
        if 'delay_penalty' in self.results:
            ax8 = plt.subplot(3, 3, 8)
            df = self.results['delay_penalty']
            ax8.plot(df['alpha'], df['total_cost'], 'o-', linewidth=2, markersize=8, label='Total Cost')
            ax8.set_xlabel('Delay Penalty Coefficient (α)', fontsize=12)
            ax8.set_ylabel('Total Cost', fontsize=12)
            ax8.set_title('Impact of Delay Penalty', fontsize=14, fontweight='bold')
            ax8.grid(True, alpha=0.3)
            
            # 添加延迟时间的双Y轴
            ax8_twin = ax8.twinx()
            ax8_twin.plot(df['alpha'], df['delay_time'], '^-', color='green', 
                        linewidth=2, markersize=6, alpha=0.7, label='Avg Delay Time')
            ax8_twin.set_ylabel('Avg Delay Time', fontsize=12, color='green')
            ax8_twin.tick_params(axis='y', labelcolor='green')

        # 9. 综合性能对比
        if len(self.results) > 0:
            ax9 = plt.subplot(3, 3, 9)
            # 为所有分析创建归一化的性能指标对比
            for key, df in self.results.items():
                if 'unserved_nodes' in df.columns:
                    label = key.replace('_', ' ').title()
                    normalized_unserved = df['unserved_nodes'] / df['unserved_nodes'].max()
                    ax9.plot(df.index, normalized_unserved, 'o-', linewidth=2, 
                            markersize=6, label=label)
            
            ax9.set_xlabel('Configuration Index', fontsize=12)
            ax9.set_ylabel('Normalized Unserved Nodes', fontsize=12)
            ax9.set_title('Performance Comparison', fontsize=14, fontweight='bold')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
     
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(save_dir, 'sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'sensitivity_analysis.pdf'), bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Plots saved to {save_dir}/")
    
    def save_results(self, save_dir='evaluation_results'):
        """保存所有结果到CSV和JSON"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存每个分析的CSV
        for key, df in self.results.items():
            csv_path = os.path.join(save_dir, f'{key}_results.csv')
            df.to_csv(csv_path, index=False)
            print(f"📁 Saved {key} results to {csv_path}")
        
        # 保存汇总的JSON
        summary = {}
        for key, df in self.results.items():
            summary[key] = df.to_dict('records')
        
        json_path = os.path.join(save_dir, 'all_results.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📁 Saved summary to {json_path}")
    
    def generate_latex_tables(self, save_dir='evaluation_results'):
        """生成LaTeX格式的表格"""
        os.makedirs(save_dir, exist_ok=True)
        
        for key, df in self.results.items():
            # 选择重要的列
            important_cols = ['total_cost', 'service_rate', 'travel_cost', 'delay_penalty']
            if 'total_episode_time' in df.columns:            # ☆ 新增：若存在就展示
                important_cols.insert(1, 'total_episode_time')
            available_cols = [col for col in important_cols if col in df.columns]
            
            # 格式化数据
            latex_df = df[list(df.columns[:1]) + available_cols].round(2)
            
            # 生成LaTeX表格
            latex_str = latex_df.to_latex(index=False, caption=f"{key.replace('_', ' ').title()} Analysis Results")
            
            # 保存到文件
            latex_path = os.path.join(save_dir, f'{key}_table.tex')
            with open(latex_path, 'w') as f:
                f.write(latex_str)
            
            print(f"📝 Saved LaTeX table to {latex_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Evaluate trained MADDPG models")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of episodes per evaluation')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    
    # 分析选项
    parser.add_argument('--analyze_capacity', action='store_true',
                       help='Analyze drone capacity impact')
    parser.add_argument('--analyze_battery', action='store_true',
                       help='Analyze battery capacity impact')
    parser.add_argument('--analyze_speed', action='store_true',
                       help='Analyze speed ratio impact')
    parser.add_argument('--analyze_dynamic', action='store_true',
                       help='Analyze degree of dynamism impact')
    parser.add_argument('--analyze_cost_ratio', action='store_true',
                       help='Analyze travel cost ratio impact')
    parser.add_argument('--analyze_delay_penalty', action='store_true',
                       help='Analyze delay penalty coefficient impact')
    parser.add_argument('--analyze_all', action='store_true',
                       help='Run all sensitivity analyses')

    # 网格评估开关
    parser.add_argument('--analyze_damage_dod', action='store_true',
                        help='Analyze grid of [road_damage_ratio × DoD] with fixed training n')

    # 列表参数（空格分隔）
    parser.add_argument('--damages', nargs='+', type=float, default=[0.1, 0.3, 0.5],
                        help='List of road damage ratios (lambda)')
    parser.add_argument('--dod_values', nargs='+', type=float, default=[0.2, 0.5, 0.7],
                        help='List of DoD values to test')
    
    args = parser.parse_args()
    
    # 打印开始信息
    print("=" * 60)
    print("🚀 MADDPG Model Evaluation and Sensitivity Analysis")
    print("=" * 60)
    print(f"📂 Checkpoint Directory: {args.checkpoint_dir}")
    print(f"🎮 Episodes per Test: {args.episodes}")
    print(f"💾 Save Directory: {args.save_dir}")
    print("=" * 60)
    # input("\n按Enter键开始评估设置...")
    
    # 从配置文件读取参数
    env_cfg = cfg['environment']
    num_trucks = env_cfg['num_trucks']
    num_drones = env_cfg['num_drones']
    total_nodes = env_cfg['num_depots'] + env_cfg['num_customers']
    
    # 创建评估器
    print("\n📚 Loading trained models...")
    # input("按Enter键加载模型...")
    evaluator = ModelEvaluator(args.checkpoint_dir, num_trucks, num_drones, total_nodes)
    
    # 创建敏感性分析器
    analyzer = SensitivityAnalyzer(evaluator, cfg)
    
    # 运行基准测试
    print("\n🎯 Running baseline evaluation...")
    print("\n🎯 运行基准评估...")
    # input("按Enter键开始基准评估...")
    baseline_metrics = analyzer.run_evaluation(num_episodes=args.episodes, render=args.render)
    
    print("\n📊 Baseline Results:")
    print("-" * 40)
    for key, value in baseline_metrics.items():
        if key.startswith('avg_'):
            print(f"  {key[4:]:20s}: {value:10.2f}")
    
    # input("\n按Enter键进行敏感性分析...")
    # 保存基线的均值/方差成一行摘要
    import os, pandas as pd, json
    os.makedirs(args.save_dir, exist_ok=True)

    avg_row = {k.replace('avg_', ''): v for k, v in baseline_metrics.items() if k.startswith('avg_')}
    std_row = {k.replace('std_', '') + '_std': v for k, v in baseline_metrics.items() if k.startswith('std_')}
    summary = {**avg_row, **std_row, 'episodes': args.episodes}

    pd.DataFrame([summary]).to_csv(os.path.join(args.save_dir, 'baseline_summary.csv'), index=False)
    with open(os.path.join(args.save_dir, 'baseline_summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📁 Saved baseline summary to {os.path.join(args.save_dir, 'baseline_summary.csv')}")
    # 运行敏感性分析
    if args.analyze_all or args.analyze_cost_ratio:
        print("\n" + "="*60)
        print("开始旅行成本比分析")
        print("="*60)
        # input("按Enter键开始...")
        analyzer.analyze_cost_ratio()
    
    if args.analyze_all or args.analyze_delay_penalty:
        print("\n" + "="*60)
        print("开始延迟惩罚系数分析")
        print("="*60)
        # input("按Enter键开始...")
        analyzer.analyze_delay_penalty()

    if args.analyze_all or args.analyze_capacity:
        print("\n" + "="*60)
        print("开始无人机载重容量分析")
        print("="*60)
        # input("按Enter键开始...")
        analyzer.analyze_drone_capacity()
    
    if args.analyze_all or args.analyze_battery:
        print("\n" + "="*60)
        print("开始电池容量分析")
        print("="*60)
        # input("按Enter键开始...")
        analyzer.analyze_battery_capacity()
    
    if args.analyze_all or args.analyze_speed:
        print("\n" + "="*60)
        print("开始速度比分析")
        print("="*60)
        # input("按Enter键开始...")
        analyzer.analyze_speed_ratio()
    
    if args.analyze_all or args.analyze_dynamic:
        print("\n" + "="*60)
        print("开始动态度分析")
        print("="*60)
        # input("按Enter键开始...")
        analyzer.analyze_dynamic_degree()

    if args.analyze_all or args.analyze_damage_dod:
        print("\n" + "="*60)
        print("开始 [λ × DoD] 网格评估（固定训练 n）")
        print("="*60)
        # input("按Enter键开始...")
        analyzer.analyze_damage_dod(
            damage_ratios=args.damages,
            dod_values=args.dod_values,
            episodes=args.episodes
        )
    
    # 保存和可视化结果
    if len(analyzer.results) > 0:
        print("\n📈 生成图表并保存结果...")
        # input("按Enter键生成可视化...")
        analyzer.plot_all_results(args.save_dir)
        analyzer.save_results(args.save_dir)
        analyzer.generate_latex_tables(args.save_dir)
    
    print("\n✅ 评估成功完成!")
    print(f"📁 结果保存到: {args.save_dir}")
    # input("按Enter键退出...")

    # 清理资源
    import gc
    gc.collect()
    
    # 如果使用了TensorFlow，清理会话
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except:
        pass

if __name__ == "__main__":
    main()
