import gym
import numpy as np
from datetime import datetime
from gym import spaces
from trucks_and_drones.config import cfg

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            name: str,
            simulation,
            visualizer,
            obs_encoder,
            act_decoder,
            reward_calc,
    ):
        super().__init__()
        # 给环境一个唯一名字
        self.name = name + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
        # 保存各个组件
        self.simulation   = simulation
        self.visualizer   = visualizer
        self.obs_encoder  = obs_encoder
        self.act_decoder  = act_decoder
        self.reward_calc  = reward_calc
        self.temp_db = simulation.temp_db
        self.max_steps = max(1, int(getattr(
            self.temp_db, 'max_steps', cfg.get('environment', {}).get('max_steps', 1000)
        )))
        
        # 修改：从 temp_db 直接获取卡车和无人机数量
        self.num_trucks = getattr(self.temp_db, 'num_trucks')
        self.num_drones = getattr(self.temp_db, 'num_drones')
        self.total_agents = self.num_trucks + self.num_drones
        
        print(f"Environment initialized with {self.num_trucks} trucks and {self.num_drones} drones")
        # 确保 temp_db 字段完整性
        self._ensure_temp_db_fields()

        # 完成动作空间的构建（原始可能是 Tuple of Discrete…）
        self.act_decoder.finish_init()
        first_obs = self.reset()
        
        # 构建动作空间：为每个智能体（卡车+无人机）创建独立的动作空间
        raw_aspace = self.act_decoder.action_space()
        
        if hasattr(raw_aspace, 'spaces') and len(raw_aspace.spaces) > 0:
            # 验证动作空间数量是否匹配
            expected_agents = self.num_trucks + self.num_drones
            # if len(raw_aspace.spaces) != expected_agents:
                # print(f"Warning: Action space count mismatch. Expected {expected_agents}, got {len(raw_aspace.spaces)}")
            
            self.action_space = raw_aspace
        else:
            raise ValueError("Invalid action space structure")

        # 构造 observation_space
        if self.obs_encoder.output_as_array:
            # 如果输出为单一数组
            if hasattr(first_obs, 'shape'):
                dim = first_obs.shape[0]
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(dim,), dtype=np.float32)
            else:
                raise ValueError("Invalid observation format for array output")
        else:
            # 如果输出为 (agent_obs, global_obs) 元组
            self.observation_space = self.obs_encoder.obs_space()

        # print("Action space:", self.action_space)
        # print("Observation space:", self.observation_space)


    def _ensure_temp_db_fields(self):
        """确保 temp_db 中有必要的字段"""
        try:
            # 检查是否已有 v_coord 字段
            if 'v_coord' not in self.temp_db.status_dict:
                # print("🔧 Initializing v_coord field in temp_db")
                num_vehicles = self.num_trucks + self.num_drones
                # 初始化所有车辆都在 depot (节点 0)
                self.temp_db.status_dict['v_coord'] = np.zeros(num_vehicles, dtype=int)

            # 检查卡车相关字段
            truck_fields = ['ET', 'LT', 'NT', 'TW']
            for field in truck_fields:
                if field not in self.temp_db.status_dict:
                    # print(f"Initializing truck field '{field}' in temp_db")
                    if field == 'TW':
                        self.temp_db.status_dict[field] = np.full((self.num_trucks,), self.temp_db.WT_max, dtype=float)
                    else:
                        self.temp_db.status_dict[field] = np.zeros((self.num_trucks,), dtype=int if field != 'LT' else float)

            # 检查无人机相关字段
            drone_fields = ['ED', 'LD', 'ND', 'DW', 'attached_truck']
            for field in drone_fields:
                if field not in self.temp_db.status_dict:
                    # print(f"Initializing drone field '{field}' in temp_db")
                    self.temp_db.status_dict[field] = np.zeros((self.num_drones,), dtype=int if field != 'LD' and field != 'DW' else float)

            # 检查其他必要字段
            if 'delta' not in self.temp_db.status_dict:
                self.temp_db.status_dict['delta'] = np.ones((self.temp_db.num_nodes,), dtype=int)

            # 初始化 visited_nodes 集合（如果不存在）
            if not hasattr(self.temp_db, 'visited_nodes'):
                self.temp_db.visited_nodes = set()
                # print("🔧 Initialized visited_nodes set")

            # print("✅ temp_db fields verification completed")

        except Exception as e:
            # print(f"❌ Error initializing temp_db fields: {e}")
            # 创建一个基本的初始化
            num_vehicles = self.num_trucks + self.num_drones
            if not hasattr(self.temp_db, 'status_dict'):
                self.temp_db.status_dict = {}
            self.temp_db.status_dict['v_coord'] = np.zeros(num_vehicles, dtype=int)
            self.temp_db.visited_nodes = set()

    # def get_mask(self):
    #     """生成动作掩码，排除未激活的动态节点"""
    #     # 获取节点状态
    #     delta = self.temp_db.get_val('delta')
        
    #     # 获取道路损坏信息（如果存在）
    #     if hasattr(self.temp_db, 'road_damaged'):
    #         damaged = self.temp_db.road_damaged
    #     else:
    #         damaged = [False] * len(delta)
        
        
    #     # 只有delta=1且未受损的节点才能被选择
    #     active_unvisited = delta == 1
    #     not_damaged = np.logical_not(damaged)
    #     mask = np.logical_and(active_unvisited, not_damaged)
        
    #     # 调试输出
    #     active_indices = np.where(delta == 1)[0]
    #     inactive_indices = np.where(delta == -1)[0]
    #     visited_indices = np.where(delta == 0)[0]
        
    #     print(f"🎭 Generated mask with fixed node space:")
    #     print(f"   Active unvisited nodes (delta=1): {active_indices}")
    #     print(f"   Visited nodes (delta=0): {visited_indices}")
    #     print(f"   Inactive dynamic nodes (delta=-1): {inactive_indices}")
    #     print(f"   Final mask: {np.where(mask)[0]}")
        
    #     return mask.astype(np.bool_)

    def get_mask(self, agent_index=None):
        """
        生成动作掩码，排除未激活的动态节点
        对于卡车：额外排除道路受损节点
        对于无人机：可以访问道路受损节点
        
        Args:
            agent_index: 智能体索引，用于区分卡车和无人机
        """
        # 获取节点状态
        delta = self.temp_db.get_val('delta')
        
        # 基础掩码：只有delta=1的节点才能被选择（激活且未访问）
        mask = delta == 1
        
        # 如果提供了智能体索引，根据类型应用不同的掩码策略
        if agent_index is not None:
            # 判断是否为卡车（前num_trucks个智能体是卡车）
            is_truck = agent_index < self.num_trucks
            
            if is_truck:
                # 卡车需要额外排除道路受损节点
                if hasattr(self.temp_db, 'get_road_damaged_nodes'):
                    damaged_nodes = set(self.temp_db.get_road_damaged_nodes())
                    # 将受损节点在掩码中设为False
                    for node_idx in damaged_nodes:
                        if node_idx < len(mask):
                            mask[node_idx] = False
                    # print(f"🚧 Truck {agent_index} mask excludes {len(damaged_nodes)} damaged nodes: {damaged_nodes}")
                
                # print(f"🚛 Generated mask for TRUCK {agent_index}")
            else:
                # 无人机可以访问所有激活的节点（包括道路受损节点）
                drone_idx = agent_index - self.num_trucks
                # print(f"🚁 Generated mask for DRONE {drone_idx} (can access damaged roads)")
        
        # 调试输出
        active_indices = np.where(delta == 1)[0]
        masked_indices = np.where(mask)[0]
        
        # print(f"🎭 Mask generation:")
        # print(f"   Total active nodes (delta=1): {active_indices}")
        # print(f"   Accessible nodes after mask: {masked_indices}")
        
        return mask.astype(np.bool_)

    def reset(self):
        # Get statistics BEFORE resetting (if not first episode)
        if hasattr(self, 'count_episodes') and self.count_episodes > 0:
            # Save the previous episode's statistics before resetting
            if hasattr(self.reward_calc, 'get_episode_statistics'):
                self.last_episode_stats = self.reward_calc.get_episode_statistics()
            
        # 切换 episode
        self.count_episodes = getattr(self, 'count_episodes', 0) + 1
        self.count_steps = 0

        # 重置所有模块
        self.act_decoder.reset()
        self.obs_encoder.reset()
        self.simulation.reset()
        self.visualizer.reset()

        # ==================== 【新增部分】 ====================
        # 评估配置提供数据集时重载标准坐标；训练配置为空时保留随机坐标。
        
        dataset_path = cfg['environment'].get('dataset_path')
        if dataset_path:
            self.temp_db.apply_dataset_and_damage(
                dataset_path,
                cfg['environment']
            )
            # ==================== 【新增这一段】 ====================
            # 只要环境一重置加载完新地图，立刻自动导出 JSON 供 MIP 使用！
            from trucks_and_drones.simulation.temp_database import export_instance_coords
            export_instance_coords(self.temp_db, path="rl_instance_coords.json")
            # ======================================================
        # ====================================================

        # Only reset reward calculator AFTER first episode
        if self.count_episodes > 1:  # Don't reset on the very first episode
            if hasattr(self.reward_calc, 'reset'):
                self.reward_calc.reset()
            elif hasattr(self.reward_calc, 'funcs'):
                self.reward_calc.funcs.reset_episode_costs()
    
        # 重置 agent 轮到
        self.cur_agent = 0

        obs_n, global_obs = self.obs_encoder.observe_state()
        return (obs_n, global_obs)


    def step(self, action_n):
        """
        修复：支持多智能体同时决策和执行，使用完整的decode协调逻辑
        action_n: 长度为 num_trucks + num_drones 的动作列表
        """
        print(f"\n🎬 Environment Step {self.count_steps}")
        # print(f"   Received actions: {action_n}")
        # # 打印当前状态信息
        # print(f"   Current state before action:")
        # print(f"     Vehicle coords: {self.temp_db.status_dict.get('v_coord', 'N/A')}")
        # print(f"     ET (truck status): {self.temp_db.status_dict.get('ET', 'N/A')}")
        # print(f"     ED (drone status): {self.temp_db.status_dict.get('ED', 'N/A')}")
        # print(f"     Delta (unvisited): {self.temp_db.status_dict.get('delta', 'N/A')}")

        # 验证动作数量
        expected_actions = self.num_trucks + self.num_drones
        # if len(action_n) != expected_actions:
        #     print(f"Warning: Expected {expected_actions} actions, got {len(action_n)}")

        # **关键修复：使用完整的decode方法处理所有智能体动作协调**
        step_error = None
        try:
            # 直接使用act_decoder的decode方法，它包含完整的汇合逻辑
            all_valid_actions = self.act_decoder.decode(action_n)
            # print(f"   All valid actions from decoder: {all_valid_actions}")
        except Exception as e:
            print(f"   ❌ Action decoding failed; terminating episode: {e!r}", flush=True)
            step_error = ('action_decode_error', repr(e))
            all_valid_actions = {}

        # 一次性执行所有有效动作
        try:
            if all_valid_actions:
                _, r, done, info = self.simulation.step(all_valid_actions)
            else:
                # 如果没有有效动作，传递空动作但仍推进仿真
                # print("   ⚠️  No valid actions, executing empty step")
                _, r, done, info = self.simulation.step({})
        except Exception as e:
            print(f"   ❌ Simulation step failed; terminating episode: {e!r}", flush=True)
            step_error = step_error or ('simulation_step_error', repr(e))
            r, done, info = 0.0, False, {}

        # # 打印执行后的状态
        # print(f"   State after action execution:")
        # print(f"     Vehicle coords: {self.temp_db.status_dict.get('v_coord', 'N/A')}")
        # print(f"     ET (truck status): {self.temp_db.status_dict.get('ET', 'N/A')}")
        # print(f"     ED (drone status): {self.temp_db.status_dict.get('ED', 'N/A')}")

        # 获取新观测
        try:
            obs_result = self.obs_encoder.observe_state()
        except Exception as e:
            # print(f"   ❌ Observation encoding failed: {e}")
            # 返回之前的观测或默认观测
            obs_result = self.obs_encoder.observe_state()

        # 修复：先检查done状态，再增加step计数
        # if done:
            # print(f"   🎯 Terminal condition reached at step {self.count_steps}")
        # 在返回前增加step计数，这样episode length就是正确的
        self.count_steps += 1

        # Enforce the configured physical decision-step cap.  Previously
        # config['environment']['max_steps'] was never consulted.
        if step_error is not None:
            done = True
            if not isinstance(info, dict):
                info = {}
            info['termination_reason'], info['error'] = step_error
        elif self.count_steps >= self.max_steps and not done:
            done = True
            if not isinstance(info, dict):
                info = {}
            info['termination_reason'] = 'max_steps'
            print(
                f"   ⚠️ Episode force-terminated at max_steps={self.max_steps}",
                flush=True
            )

        # # 返回每个agent相同的奖励（合作设置）
        # n = self.total_agents
        # rew_n = [float(r)] * n
        
        # --- 规范化奖励为“按智能体的向量” ---
        n = self.total_agents
        # r 可能是标量（旧逻辑）或向量（新 per-agent 奖励）
        if np.isscalar(r):
            rew_n = [float(r)] * n
        else:
            r_arr = np.asarray(r, dtype=np.float32).reshape(-1)
            if r_arr.size == n:
                # 已经是按智能体拆分的奖励
                rew_n = r_arr.tolist()
            elif r_arr.size == 1:
                # 单元素数组，视作标量广播
                rew_n = [float(r_arr.item())] * n
            else:
                raise ValueError(f"[Env.step] reward shape mismatch: got {r_arr.shape}, expected {n} or 1")
        # 把按智能体的奖励也透传给上层，便于训练侧直接使用
        if isinstance(info, dict):
            info['agent_step_rewards'] = np.asarray(rew_n, dtype=np.float32)
            

        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            agent_obs, global_obs = obs_result
            # print(f"   Step result: reward={r}, done={done}, agent_obs_count={len(agent_obs)}")
            return (agent_obs, global_obs), rew_n, done, info
        else:
            # print(f"   Step result: reward={r}, done={done}, obs_shape={obs_result.shape if hasattr(obs_result, 'shape') else 'unknown'}")
            return obs_result, rew_n, done, info


    def get_valid_action_mask(self, agent_index, obs):
        """
        返回一个一维 0/1 向量 mask，长度等于 action_space.n，
        mask[a]=1 表示动作 a 在当前状态下合法，否则为 0。
        (i) 访问过的节点必须屏蔽 —— 用 temp_db.visited_nodes
        (ii) 其他约束（容量、电量、可达性等）也在这里过滤。
        """
        A = self.action_space[agent_index].n
        mask = np.zeros(A, dtype=np.float32)

        # (i) Node Visitation Masking
        visited = self.temp_db.visited_nodes

        # (ii) Constraint‐Based Filtering: 举例
        truck_cap = self.temp_db.status_dict['TW'][agent_index]
        drone_bat = self.temp_db.status_dict.get('battery', np.zeros_like(truck_cap))[agent_index]
        accessible = set(self.temp_db.get_unvisited_nodes()) - set(self.temp_db.get_road_damaged_nodes())

        # 这里假设 get_unvisited_nodes 只返回未访问，get_road_damaged_nodes 返回受损节点

        for a in range(A):
            # 先排除已访问
            if a in visited:
                continue
            # 检查路损、可达性
            if a not in accessible:
                continue
            # 检查卡车载重
            if self.temp_db.get_val('demand')[a] > truck_cap:
                continue
            # 检查无人机电量（简单示例，来回两倍距离）
            try:
                current_pos = self.temp_db.status_dict['v_coord'][agent_index]
                dist = self.temp_db.distance(current_pos, a)
                if 2 * dist > drone_bat:
                    continue
            except:
                # 如果距离计算失败，跳过这个动作
                continue
            # 如果还有其他约束（绑定、水量、等待等），都可以加在这里
            mask[a] = 1.0

        return mask

    def render(self, mode='human', close=False, slow_down_pls=False):
        if mode == 'human':
            self.visualizer.visualize_step(
                self.count_episodes,
                self.count_steps,
                slow_down_pls
            )
        if close:
            self.visualizer.close()
