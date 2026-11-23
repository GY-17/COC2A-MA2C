import numpy as np
from trucks_and_drones.config import cfg   # 新增：从 config 读取训练参数

def reward_parameter(
    reward_modes        = None,
    reward_type         = 'sum_vehicle',
    restriction_rewards = ['travel_cost', 'delay_penalty'],
    action_rewards      = ['service_reward'],
):
    """
    保持与 build_env.py 接口一致，但实际计算时我们
    只使用了 service_reward、travel_cost 和 delay_penalty。
    """
    return {
        'reward_modes'       : reward_modes,
        'reward_type'        : reward_type,
        'restriction_rewards': restriction_rewards,
        'action_rewards'     : action_rewards,
    }


class RewardFunctions:
    """
    按论文公式计算本步系统即时奖励：
      r_t = sum_{k ∈ U_t} [ e^k − (c^k + d^k) ]

    修复内容：
    1. 正确计算服务奖励和旅行成本
    2. 修复延迟惩罚计算
    3. 考虑实际到达时间而不是当前时间
    """

    def __init__(self, temp_db):
        self.db = temp_db
        # 假定这些参数预先保存在 temp_db
        self.ct = temp_db.ct_cost       # 卡车单位时间成本 c_t
        self.cd = temp_db.cd_cost       # 无人机单位时间成本 c_d
        self.vt = temp_db.truck_speed   # 卡车速度 v_t
        self.vd = temp_db.drone_speed   # 无人机速度 v_d

        # === 从 config.training 读取成本权重（没有配置时，用默认值）===
        train_cfg = cfg.get('training', {})
        self.travel_weight = float(train_cfg.get('reward_travel_weight', 2.0))
        self.delay_weight  = float(train_cfg.get('reward_delay_weight', 1.0))
        self.reward_scale  = float(train_cfg.get('reward_scale', 1.0))  # 新增

        print(
            f"   💲 Cost coefficients: ct={self.ct}, cd={self.cd}; "
            f"speeds: truck={self.vt}, drone={self.vd}; "
            f"reward_scale={self.reward_scale} (training reward only)"
        )
        print(f"   ⚖️ Reward weights: travel={self.travel_weight}, delay={self.delay_weight}")


        # 新增：累积成本统计（每个episode重置）
        self.reset_episode_costs()
        print(f"   📊 Initialized episode cost tracking")

    def reset_episode_costs(self):
        """重置每个episode的成本统计"""
        self.episode_travel_cost = 0.0
        self.episode_delay_penalty = 0.0
        self.episode_truck_travel_time = 0.0
        self.episode_drone_travel_time = 0.0
        # 新增：统计“服务收益总和”（不带任何成本权重）
        self.episode_service_reward = 0.0
        self.served_nodes = set()  # 已服务节点
        # 确保 temp_db 里的统计也被重置
        if hasattr(self.db, 'stats_truck_move_time'):
            self.db.stats_truck_move_time = 0.0
            self.db.stats_truck_wait_time = 0.0
            self.db.stats_drone_move_time = 0.0
            self.db.stats_drone_wait_time = 0.0
    
    def get_episode_statistics(self):
        """获取当前episode的统计数据"""
        # 计算未服务节点数
        delta = self.db.get_val('delta')
        beta = self.db.get_val('beta')  # 获取每个节点的beta值
        
        # 找出所有未服务的客户节点
        unserved_nodes = []
        for i, d in enumerate(delta):
            if i > 0:  # 排除depot (index 0)
                if d == 1:  # delta=1 表示激活但未服务
                    unserved_nodes.append(i)
        
        unserved_count = len(unserved_nodes)

        # 未服务惩罚：每个未服务节点的惩罚等于其beta值
        unserved_penalty = sum(beta[node_idx] for node_idx in unserved_nodes)
     
        # 总成本
        total_cost = self.episode_travel_cost + self.episode_delay_penalty + unserved_penalty
        # 新增：真实总 reward = 服务收益总和 - 原始成本
        real_reward = self.episode_service_reward - total_cost
    
        # 【修改】获取细分数据
        t_move = getattr(self.db, 'stats_truck_move_time', 0.0)
        t_wait = getattr(self.db, 'stats_truck_wait_time', 0.0)
        d_move = getattr(self.db, 'stats_drone_move_time', 0.0)
        d_wait = getattr(self.db, 'stats_drone_wait_time', 0.0) # <--- 【新增】读取无人机等待时间

        return {
            'total_cost': float(total_cost),
            'travel_cost': float(self.episode_travel_cost),
            'delay_penalty': float(self.episode_delay_penalty),
            'unserved_penalty': float(unserved_penalty),
            'unserved_count': int(unserved_count),
            'served_count': int(len(self.served_nodes)),
            # 原始细分数据
            'truck_move_time': float(t_move),
            'truck_wait_time': float(t_wait),
            'drone_move_time': float(d_move),
            'drone_wait_time': float(d_wait), # <--- 【新增】输出无人机等待时间
            
            # 组合指标 (如果你需要总时间)
            'truck_travel_time': float(t_move + t_wait),
            'drone_travel_time': float(d_move + d_wait), # <--- 【修改】现在你可以选择包含或不包含等待
            # 新增两个统计量
            'real_reward': float(real_reward),                    # 服务收益 − 原始成本
            'service_reward_sum': float(self.episode_service_reward),  # 纯服务收益
        }

    def per_agent_reward(self, actions: dict):
        """
        返回 shape=(num_trucks+num_drones,) 的向量，每个元素是该智能体本步的净奖励。
        仍然会更新 self.episode_travel_cost / self.episode_delay_penalty 等统计。
        """
        t = self.db.total_time
        D     = self.db.get_val('deadline')
        alpha = self.db.get_val('alpha')
        beta  = self.db.get_val('beta')

        num_trucks = getattr(self.db, 'num_trucks', 0)
        num_drones = getattr(self.db, 'num_drones', 0)
        n_agents = num_trucks + num_drones
        r_by_agent = np.zeros(n_agents, dtype=np.float32)

        for agent_idx, act in actions.items():
            if not act:
                continue

            if agent_idx < num_trucks:
                # ---------- 卡车 ----------
                truck_idx = agent_idx
                truck_pos = self.db.status_dict.get('v_coord', [0]*num_trucks)[truck_idx]

                tgt = act.get('truck_target_node')
                # The decoder may leave a target alongside truck_wait.  A
                # waiting truck does not travel to that target in this step.
                if tgt is not None and act.get('truck_wait', 0) != 1:
                    if tgt == 0:
                        # 回库：只有路程成本
                        # dist = self.db.distance(truck_pos, 0)
                        dist = self.db.distance(truck_pos, 0, metric='manhattan')
                        travel_time = dist / self.vt
                        travel_cost = self.ct * travel_time
                        self.episode_travel_cost += travel_cost
                        self.episode_truck_travel_time += travel_time
                        # r_by_agent[agent_idx] -= travel_cost
                        r_by_agent[agent_idx] -= self.travel_weight * travel_cost
                    elif tgt != 0:
                        # 服务：服务奖励 - 旅行成本 - 延迟惩罚
                        # dist = self.db.distance(truck_pos, tgt)
                        dist = self.db.distance(truck_pos, tgt, metric='manhattan')
                        travel_time = dist / self.vt
                        arrival_time = t + travel_time

                        service_reward = beta[tgt]
                        travel_cost    = self.ct * travel_time
                        delay_penalty  = alpha[tgt] * max(0.0, arrival_time - D[tgt])

                        self.episode_travel_cost += travel_cost
                        self.episode_delay_penalty += delay_penalty
                        self.episode_truck_travel_time += travel_time
                        self.served_nodes.add(tgt)
                        
                        # 新增：累计“真实服务收益”（不加权）
                        self.episode_service_reward += service_reward

                        r_by_agent[agent_idx] += service_reward - self.travel_weight * travel_cost - self.delay_weight  * delay_penalty

                if act.get('truck_wait', 0) == 1:
                    wait_cost = self.ct * 0.1
                    self.episode_travel_cost += wait_cost
                    r_by_agent[agent_idx] -= self.travel_weight * wait_cost

            else:
                # ---------- 无人机 ----------
                drone_idx = agent_idx - num_trucks
                drone_pos = self._get_current_drone_pos(drone_idx)

                svc = act.get('drone_service_node')
                # drone_continue means no service flight was executed.
                if svc is not None and svc != 0 and act.get('drone_continue', 0) != 1:
                    # dist = self.db.distance(drone_pos, svc)
                    dist = self.db.distance(drone_pos, svc, metric='euclidean')
                    travel_time = dist / self.vd
                    arrival_time = t + travel_time

                    service_reward = beta[svc]
                    travel_cost    = self.cd * travel_time
                    delay_penalty  = alpha[svc] * max(0.0, arrival_time - D[svc])

                    self.episode_travel_cost += travel_cost
                    self.episode_delay_penalty += delay_penalty
                    self.episode_drone_travel_time += travel_time
                    self.served_nodes.add(svc)

                    # 新增：累计服务收益
                    self.episode_service_reward += service_reward
                   
                    r_by_agent[agent_idx] += service_reward - self.travel_weight * travel_cost - self.delay_weight  * delay_penalty

                # 汇合：只有路程成本
                if 'drone_rendezvous_node' in act and act['drone_rendezvous_node'] is not None:
                    node = act['drone_rendezvous_node']
                    # dist = self.db.distance(drone_pos, node)
                    dist = self.db.distance(drone_pos, node, metric='euclidean')
                    travel_time = dist / self.vd
                    travel_cost = self.cd * travel_time
                    self.episode_travel_cost += travel_cost
                    self.episode_drone_travel_time += travel_time
                    r_by_agent[agent_idx] -= self.travel_weight * travel_cost

                # 继续飞行/等待不额外给奖励，这里可按需塑形
                # if act.get('drone_continue', 0) == 1: pass

        # return r_by_agent
        # ✅ 只在输出时做统一缩放
        return r_by_agent / self.reward_scale

    def system_reward(self, actions: dict) -> float:
        """
        系统总奖励 = 各智能体奖励之和。为兼容旧接口，这里只做求和，
        具体的统计已在 per_agent_reward() 内完成，避免重复累计。
        """
        r_by_agent = self.per_agent_reward(actions)
        return float(np.sum(r_by_agent))
    

    # def system_reward(self, actions: dict) -> float:
    #     """
    #     修复后的系统奖励计算 - 支持独立卡车和无人机智能体
    #     """
    #     # 当前仿真时刻
    #     t = self.db.total_time
    #     print(f"   💰 Reward calculation at time {t:.3f}")

    #     # 2) 节点属性
    #     D     = self.db.get_val('deadline')  # 每个节点的 Di
    #     alpha = self.db.get_val('alpha')     # 每个节点的 α_i
    #     beta  = self.db.get_val('beta')      # 每个节点的 β_i

    #     total_r = 0.0
        
    #     # 获取智能体数量
    #     num_trucks = getattr(self.db, 'num_trucks', 0)
    #     num_drones = getattr(self.db, 'num_drones', 0)

    #     # 遍历所有动作
    #     for agent_idx, act in actions.items():
    #         if not act:  # 空动作
    #             continue
                
    #         print(f"   💰 Calculating reward for agent {agent_idx}: {act}")
            
    #         # **关键修复：区分卡车和无人机智能体**
    #         if agent_idx < num_trucks:
    #             # 卡车智能体
    #             truck_idx = agent_idx
    #             truck_pos = self.db.status_dict.get('v_coord', [0] * num_trucks)[truck_idx]
                
    #             # 1. 卡车服务动作
    #             if 'truck_target_node' in act and act['truck_target_node'] is not None:
    #                 target_node = act['truck_target_node']
                    
    #                 if target_node > 0:  # 只有customer节点才有服务奖励
    #                     dist = self.db.distance(truck_pos, target_node)
    #                     travel_time = dist / self.vt
    #                     arrival_time = t + travel_time
                        
    #                     # 服务奖励
    #                     service_reward = beta[target_node]
                        
    #                     # 旅行成本
    #                     travel_cost = self.ct * travel_time
    #                     self.episode_travel_cost += travel_cost  # 累积
    #                     print(f"     📊 Accumulated travel cost: {self.episode_travel_cost:.2f} (added {travel_cost:.2f})")
    #                     self.episode_truck_travel_time += travel_time  # 累积卡车旅行时间                        
                        
    #                     # 延迟惩罚
    #                     delay_penalty = alpha[target_node] * max(0.0, arrival_time - D[target_node])
    #                     self.episode_delay_penalty += delay_penalty  # 累积
    #                     # 记录已服务节点
    #                     self.served_nodes.add(target_node)
                        
    #                     reward = service_reward - travel_cost - delay_penalty
    #                     total_r += reward
                        
    #                     print(f"     🚛 Truck {truck_idx} service node {target_node}: service={service_reward:.2f}, cost={travel_cost:.2f}, delay={delay_penalty:.2f}, net={reward:.2f}")
                    
    #                 elif target_node == 0:
    #                     # 返回depot只有旅行成本
    #                     dist = self.db.distance(truck_pos, target_node)
    #                     travel_time = dist / self.vt
    #                     travel_cost = self.ct * travel_time
    #                     self.episode_travel_cost += travel_cost  # 累积
    #                     print(f"     📊 Accumulated travel cost: {self.episode_travel_cost:.2f} (added {travel_cost:.2f})")
    #                     self.episode_truck_travel_time += travel_time  # 累积
    #                     total_r -= travel_cost
    #                     print(f"     🚛 Truck {truck_idx} to depot: cost={travel_cost:.2f}")

    #             # 2. 卡车等待动作
    #             if act.get('truck_wait', 0) == 1:
    #                 # 等待动作有小的时间成本
    #                 wait_cost = self.ct * 0.1  # 小的等待成本
    #                 self.episode_travel_cost += wait_cost  # 累积到旅行成本中
    #                 total_r -= wait_cost
    #                 print(f"     ⏳ Truck {truck_idx} waiting: cost={wait_cost:.2f}")
                    
    #         else:
    #             # 无人机智能体
    #             drone_idx = agent_idx - num_trucks
    #             drone_pos = self._get_current_drone_pos(drone_idx)
                
    #             # 3. 无人机服务动作
    #             if 'drone_service_node' in act and act['drone_service_node'] is not None:
    #                 target_node = act['drone_service_node']
    #                 if target_node > 0:  # 只有customer节点才有服务奖励
    #                     dist = self.db.distance(drone_pos, target_node)
    #                     travel_time = dist / self.vd
    #                     arrival_time = t + travel_time
                        
    #                     # 服务奖励
    #                     service_reward = beta[target_node]
                        
    #                     # 旅行成本
    #                     travel_cost = self.cd * travel_time
    #                     self.episode_travel_cost += travel_cost  # 累积
    #                     print(f"     📊 Accumulated travel cost: {self.episode_travel_cost:.2f} (added {travel_cost:.2f})")
    #                     self.episode_drone_travel_time += travel_time  # 累积无人机旅行时间
                        
    #                     # 延迟惩罚
    #                     delay_penalty = alpha[target_node] * max(0.0, arrival_time - D[target_node])
    #                     self.episode_delay_penalty += delay_penalty  # 累积
                        
    #                     # 记录已服务节点
    #                     self.served_nodes.add(target_node)
                        
    #                     reward = service_reward - travel_cost - delay_penalty
    #                     total_r += reward
                        
    #                     print(f"     🚁 Drone {drone_idx} service node {target_node}: service={service_reward:.2f}, cost={travel_cost:.2f}, delay={delay_penalty:.2f}, net={reward:.2f}")

    #             # 4. 无人机汇合动作 - **关键修复：只有旅行成本**
    #             if 'drone_rendezvous_node' in act and act['drone_rendezvous_node'] is not None:
    #                 target_node = act['drone_rendezvous_node']
    #                 dist = self.db.distance(drone_pos, target_node)
    #                 travel_time = dist / self.vd
    #                 travel_cost = self.cd * travel_time
    #                 self.episode_travel_cost += travel_cost  # 累积
    #                 print(f"     📊 Accumulated travel cost: {self.episode_travel_cost:.2f} (added {travel_cost:.2f})")
    #                 self.episode_drone_travel_time += travel_time  # 累积
    #                 total_r -= travel_cost
    #                 print(f"     🤝 Drone {drone_idx} rendezvous at {target_node}: cost={travel_cost:.2f}")

    #             # 5. 无人机继续跟随动作（通常无成本）
    #             if act.get('drone_continue', 0) == 1:
    #                 # 继续跟随通常没有额外成本
    #                 print(f"     🔄 Drone {drone_idx} continues (no cost)")

    #     print(f"   💰 Total system reward: {total_r:.3f}")
    #     return total_r

    def _get_current_drone_pos(self, drone_idx):
        """获取无人机当前位置 - 修复独立模式支持"""
        try:
            # **关键修复：正确处理独立模式和搭载模式**
            ED = self.db.status_dict['ED'][drone_idx]
            attached_truck = self.db.status_dict.get('attached_truck', [-1] * len(self.db.status_dict['ED']))[drone_idx]
            
            if attached_truck == -1:
                # 独立模式：使用无人机自己的坐标
                if 'drone_coord' in self.db.status_dict:
                    return self.db.status_dict['drone_coord'][drone_idx]
                else:
                    return 0  # 默认在depot
            elif ED == 3 and attached_truck >= 0:
                # 搭载模式且在卡车上：返回附着卡车的位置
                if attached_truck < len(self.db.status_dict.get('v_coord', [])):
                    return self.db.status_dict['v_coord'][attached_truck]
                else:
                    return 0
            else:
                # 其他状态（飞行中、等待等）：使用无人机独立坐标
                if 'drone_coord' in self.db.status_dict:
                    return self.db.status_dict['drone_coord'][drone_idx]
                else:
                    return 0
        except (KeyError, IndexError) as e:
            print(f"     Warning: Error getting drone {drone_idx} position: {e}")
            return 0
    

class BaseRewardCalculator:
    """
    修复后的奖励计算器
    接口与 build_env.py 保持一致：
      __init__(reward_params, temp_db)
      reward_function(actions) -> float
    """

    def __init__(self, reward_params: dict, temp_db):
        self.db = temp_db
        self.reward_params = reward_params
        # 构造内部计算器
        self.funcs = RewardFunctions(temp_db)
        print(f"   🔧 Reward calculator initialized with params: {reward_params}")

    def reset(self):  # 注意：方法名是 reset，不是 reset_episode
        """重置episode统计"""
        self.funcs.reset_episode_costs()
        print(f"   📊 Reset episode cost tracking")

    def get_episode_statistics(self):
        """获取episode统计"""
        stats = self.funcs.get_episode_statistics()
        print(f"   📊 Getting episode stats: {stats}")  # 调试输出
        return stats

    def reward_function(self, actions: dict) -> float:
        """
        修复后的奖励函数
        在 Env.step() 中被调用，传入 joint actions，返回本步系统即时 reward。
        """
        if not actions:
            print(f"   💰 No actions provided, reward = 0.0")
            return 0.0
            
        reward = self.funcs.system_reward(actions)
        
        # # 额外的奖励塑形：完成任务的额外奖励
        # try:
        #     delta = self.db.get_val('delta')
        #     unserved = sum(1 for i, d in enumerate(delta) if i > 0 and d == 1)
        #     if unserved == 0:
        #         completion_bonus = 1000.0
        #         reward += completion_bonus
        #         print(f"   🎉 Task completion bonus: {completion_bonus}")
        # except:
        #     pass
            
        return reward
    
    def reward_function_per_agent(self, actions: dict):
        """
        返回每个智能体的本步奖励向量（长度 = num_trucks + num_drones）
        """
        if not actions:
            n = getattr(self.db, 'num_trucks', 0) + getattr(self.db, 'num_drones', 0)
            return np.zeros(n, dtype=np.float32)
        return self.funcs.per_agent_reward(actions)
