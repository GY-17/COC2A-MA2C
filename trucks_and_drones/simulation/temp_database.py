'''
'''
import random
import numpy as np
from trucks_and_drones.config import cfg

'''
def lookup_db(db_dict, name_list):
    obj_list = []
    return [obj_list.append(db_dict[i]) for name in name_list]
'''

import json
import numpy as np

# def export_instance_coords(temp_db, path="rl_instance_coords.json", include_dynamic=False):
#     """导出当前实例的节点坐标（单位：km），顺序：depot在前，其后为客户。"""
#     grid_size_km = float(temp_db.area_size) / float(temp_db.grid[0])         # 每个网格对应的 km
#     # 只导出 depot + 静态客户（做静态基线时更公平）；需要动态一起导出就设 include_dynamic=True
#     N = temp_db.num_depots + (temp_db.num_customers if include_dynamic else temp_db.num_static_customers)
#     coords_km = (temp_db.status_dict['n_coord'][:N] * grid_size_km).tolist() # 网格坐标 → km坐标
#     data = {
#         "coords_km": coords_km,
#         "num_depots": int(temp_db.num_depots),
#         "num_customers": int(N - temp_db.num_depots)
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
def export_instance_coords(temp_db, path="rl_instance_coords.json", include_dynamic=False):
    """【已修改】导出坐标、需求量以及受损情况给 MIP 求解器"""
    import json
    import numpy as np
    
    grid_size_km = float(temp_db.area_size) / float(temp_db.grid[0])
    N = temp_db.num_depots + (temp_db.num_customers if include_dynamic else temp_db.num_static_customers)
    
    coords_km = (temp_db.status_dict['n_coord'][:N] * grid_size_km).tolist()
    
    # 兼容获取需求量 (demand 或 n_items)
    try:
        demands = temp_db.status_dict['demand'][:N].tolist()
    except KeyError:
        demands = temp_db.status_dict['n_items'][:N].tolist()
        
    # 【修改这里】：读取 road_damaged，如果是 1 则转为 True，0 转为 False
    raw_damage = temp_db.status_dict.get('road_damaged', np.zeros(N, dtype=int))[:N]
    is_damaged = (np.array(raw_damage) == 1).tolist()

    # 🌟 新增：导出截⽌时间 deadline
    deadlines = temp_db.constants_dict['deadline'][:N].tolist()

    data = {
        "coords_km": coords_km,
        "demands": demands,
        "is_damaged": is_damaged,
        "deadlines": deadlines,  # 🌟 将其写入 JSON
        "num_depots": int(temp_db.num_depots),
        "num_customers": int(N - temp_db.num_depots)
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"📂 实例数据已导出至 {path}，准备供 MIP 求解！")

def random_coordinates(grid):
    """生成 [1..grid[0]]×[1..grid[1]] 之间的随机整数坐标。"""
    return np.array([
        np.random.randint(0, grid[0]),
        np.random.randint(0, grid[1])
    ], dtype=float)

def insert_at_coord(dict_var, key, value, list_index, num_objs):
    """在 dict_var[key]（shape=(N,2)）的第 list_index 行插入坐标 value。"""
    if key in dict_var:
        dict_var[key][list_index] = value
    else:
        arr = np.zeros((num_objs, 2), dtype=float)
        arr[list_index] = value
        dict_var[key] = arr

def insert_at_array(dict_var, key, value, list_index, num_objs):
    """在 dict_var[key]（shape=(N,)）的第 list_index 项插入数值 value。"""
    if key in dict_var:
        dict_var[key][list_index] = value
    else:
        arr = np.zeros((num_objs,), dtype=float)
        arr[list_index] = value
        dict_var[key] = arr


def insert_at_list(dict_var, key, value, list_index, num_objs):
    """在 dict_var[key]（list 长度 = N）第 list_index 处插入 value。"""
    if key in dict_var:
        dict_var[key][list_index] = value
    else:
        lst = [None] * num_objs
        lst[list_index] = value
        dict_var[key] = lst


def append_to_array(dict_var, key, value):
    """向 dict_var[key]（一维 ndarray）尾部追加一个元素 value。"""
    if key in dict_var:
        dict_var[key] = np.append(dict_var[key], value)
    else:
        dict_var[key] = np.array([value], dtype=float)



class BaseTempDatabase:

    def __init__(self, name, grid, signal_list, debug_mode):
        # 仿真标识与参数
        self.name  = name
        self.grid  = grid
        self.signal_list = signal_list
        env_cfg = cfg.get('environment', {})
        self.horizon = env_cfg['horizon']
        # Hard cap for both normal training episodes and CMA-ES inner episodes.
        self.max_steps = max(1, int(env_cfg.get('max_steps', 1000)))
        self.area_size = env_cfg['area_size']
        self.num_trucks = env_cfg['num_trucks']
        self.num_drones = env_cfg['num_drones']
        self.num_depots = env_cfg['num_depots']
        self.num_customers = env_cfg['num_customers']
        self.debug_mode = debug_mode

        # 动态节点配置
        dynamic_cfg = env_cfg.get('dynamic_nodes', {})
        self.dynamic_enabled = dynamic_cfg.get('enable')
        self.dod = dynamic_cfg.get('dod')
        self.delta_t = dynamic_cfg.get('delta_t')

        # 计算动态和静态节点数量
        self.total_customers = env_cfg['num_customers']
        if self.dynamic_enabled:
            self.num_dynamic_nodes = int(self.dod * self.total_customers)
            self.num_static_customers = self.total_customers - self.num_dynamic_nodes
            
        else:
            self.num_dynamic_nodes = 0
            self.num_static_customers = self.total_customers
        
        # 关键修复：确保总节点数包含所有静态+动态节点空间
        self.num_customers = self.total_customers  # 总客户节点数（包括动态）
        self.num_nodes = self.num_depots + self.total_customers  # 总节点数（保持固定）
        
        # print(f"Node space allocation:")
        # print(f"  Total nodes: {self.num_nodes}")
        # print(f"  Depots: {self.num_depots} (indices 0-{self.num_depots-1})")
        # print(f"  Static customers: {self.num_static_customers} (indices {self.num_depots}-{self.num_depots+self.num_static_customers-1})")
        # print(f"  Dynamic customers: {self.num_dynamic_nodes} (indices {self.num_depots+self.num_static_customers}-{self.num_nodes-1})")
        
        # 动态节点管理
        self.dynamic_nodes_pool = []
        self.active_dynamic_nodes = set()
        self.next_check_time = self.delta_t
        self.static_customer_count = self.num_static_customers  # 静态客户节点数

        # 对象计数器
        self.num_vehicles = 0
        # -----------------------------------------
        # 论文所需成本与速度参数
        self.ct_cost = env_cfg['ct_cost']  # 卡车单位时间成本 c_t
        self.cd_cost = env_cfg['cd_cost']  # 无人机单位时间成本 c_d
        self.truck_speed = env_cfg['truck_speed']  # 卡车速度 v_t
        self.drone_speed = env_cfg['drone_speed']  # 无人机速度 v_d
        # -----------------------------------------
        # 卡车/无人机 容量限制
        self.WT_max = env_cfg['WT_max']
        self.WD_max = env_cfg['WD_max']
        self.drone_battery=env_cfg['max_charge']
        self.node_visuals = []  # BaseNodeCreator 会 append([symbol, color])
        self.vehicle_visuals = []  # BaseVehicleCreator 或 create_truck_drone_pairs 会 append(...)
        self.min_max_dict = {}  # 存放诸如 {'n_type': [0, T-1], 'v_type': [0, V-1]} 之类的范围

        # 仿真状态容器，后面 init_db 填充
        self.status_dict = {}
        self.constants_dict = {}
        self.signals_dict = {}
        # unvisited 和 visited 用于 IAM
        self._unvisited = set()
        self.visited_nodes = set()
        # 索引列表
        self.v_indices = []
        self.c_indices = []
        self.d_indices = []

        # 分组字典（用于 ObsEncoder）
        self.key_groups_dict = {
            # 'coordinates': ['v_coord','v_dest','c_coord','d_coord'],
            # 'LT_time', 'LD_time'
            'coordinates': ['v_coord', 'n_coord'],
            'binary'     : ['ET', 'ED', 'unassigned'],
            # 二只放那些二值 flag
            'values': [ 'TW', 'time', 'n_items', 'deadline'],
            # 只放需要归一化的连续量, <-- 去掉 LT_time, LD_time
            'unassigned': ['delta'],  # 用于 obs 中无指派标志
            'vehicles'  : ['ET','ED','NT','ND','truck_node','drone_node','TW','LT_time','LD_time','time'],
            'customers' : ['n_items','deadline','unassigned'],
            'depots'     : [],  # 动态添加
            'action_signals': [],  # 动态添加
        }

        # 当前执行的 tandem 索引
        self.cur_tandem_index = 0

    def export_dynamic_instance_for_gu(self, path="dynamic_instance_eval.json"):
        """
        【新增方法】：将当前 RL 环境内部生成的静态+动态混合状态，
        导出为 Gu.py 标准的 JSON 数据集格式，保证 100% 公平对比。
        """
        import json
        import numpy as np
        
        # 网格坐标转真实公里坐标
        grid_size_km = float(self.area_size) / float(self.grid[0])
        
        dataset = []
        N = self.num_nodes # 包含 Depot 和所有客户
        
        # 1. 整理动态节点池信息，方便快速查找
        dynamic_dict = {}
        if hasattr(self, 'dynamic_nodes_pool'):
            for dyn_node in self.dynamic_nodes_pool:
                dynamic_dict[dyn_node['node_idx']] = dyn_node['release_time']
                
        # 2. 读取路损情况
        raw_damage = self.status_dict.get('road_damaged', np.zeros(N, dtype=int))
        
        # 3. 构造 Depot (索引 0)
        depot_coord = self.status_dict['n_coord'][0] * grid_size_km
        dataset.append({
            "id": 0,
            "x": float(depot_coord[0]), 
            "y": float(depot_coord[1]),
            "demand": float(self.status_dict['n_items'][0]),
            "is_dynamic": False, 
            "release_time": 0.0,
            "deadline": float(self.horizon), 
            "road_damaged": 0
        })
        
        # 4. 构造所有 Customer
        for i in range(1, N):
            coord = self.status_dict['n_coord'][i] * grid_size_km
            is_dynamic = (i in dynamic_dict)
            release_time = dynamic_dict.get(i, 0.0)
            
            # 读取环境内已经生成的 deadline (Solomon公式)
            deadline = float(self.constants_dict['deadline'][i])
            is_damaged = int(raw_damage[i])
            demand = float(self.status_dict['n_items'][i])
            
            dataset.append({
                "id": i,
                "x": float(coord[0]), 
                "y": float(coord[1]),
                "demand": demand,
                "is_dynamic": is_dynamic,
                "release_time": float(release_time),
                "deadline": deadline,
                "road_damaged": is_damaged
            })
            
        output_data = {
            "metadata": {
                "total_nodes": N,
                "horizon": self.horizon,
                "area_size_km": self.area_size
            },
            "nodes": dataset
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"📂 动态实例数据已导出至 {path}，准备供 GU 求解！")

    def apply_dataset_and_damage(self, dataset_path, config):
        """【新增方法】解析数据集，缩放负数坐标，生成受损节点"""
        import os
        import numpy as np
        
        if not dataset_path:
            return

        if not os.path.exists(dataset_path):
            print(f"⚠️ 警告: 未找到数据集 {dataset_path}，将使用随机坐标。")
            return

        with open(dataset_path, 'r') as f:
            # 过滤掉 /* 开头的注释行和空行
            lines = [line.strip() for line in f.readlines() if not line.startswith('/*') and line.strip()]

        # 根据 singlecenter-1-n5.txt 格式：第0行为卡车速度，1为无人机速度，2为节点数
        # 第 3 行开始是坐标 (Depot 接着 Customers)
        coords = []
        for line in lines[3:]:
            parts = line.split()
            coords.append([float(parts[0]), float(parts[1])])

        coords = np.array(coords)
        # 截取 config 中配置的客户数量 (+1 代表 Depot)
        N = min(config['num_customers'] + 1, len(coords))
        coords = coords[:N]

        # --- 完美解决含负数坐标的核心：Min-Max 映射 ---
        min_val = np.min(coords, axis=0)
        max_val = np.max(coords, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0  # 防止除以0

        # 将点云完美等比例拉伸/压缩到你的 [0, grid_x] 和 [0, grid_y] 内部网格系中
        grid_x, grid_y = self.grid[0], self.grid[1]
        scaled_coords = (coords - min_val) / range_val * [grid_x, grid_y]

        # 强制覆盖当前随机生成的坐标
        self.status_dict['n_coord'][:N] = scaled_coords

        # --- 动态随机生成道路受损节点 ---
        road_damaged = np.zeros(len(self.status_dict['n_coord']), dtype=bool)
        num_customers = N - 1
        num_damaged = int(num_customers * config.get('road_damage_ratio', 0.2))

        if num_damaged > 0:
            # np.random.choice 从客户节点 (索引 1 到 N-1) 中随机无放回抽取
            damaged_indices = np.random.choice(range(1, N), size=num_damaged, replace=False)
            road_damaged[damaged_indices] = True

        # ✅ ADD THIS LINE: Synchronize the set used by the action mask
        self._road_damaged = set(np.where(road_damaged)[0].tolist())
        self.status_dict['road_damaged'] = road_damaged
        print(f"🗺️ 数据集已加载并缩放至 {self.area_size}km 范围! 随机设置了 {num_damaged} 个受损节点。")

    def get_road_damaged_nodes(self):
        """【新增方法】为动作掩码提供接口"""
        if 'road_damaged' not in self.status_dict:
            return []
        import numpy as np
        return np.where(self.status_dict['road_damaged'])[0].tolist()

    def _generate_dynamic_nodes(self):
        """已废弃 - 动态节点属性在init_db中直接生成"""
        pass
    def initialize_road_damage(self):
        """初始化道路损坏节点 - 从所有激活的customer节点中随机选择"""
        # 获取道路损坏比例
        env_cfg = cfg.get('environment', {})
        road_damage_ratio = env_cfg.get('road_damage_ratio')
        
        if road_damage_ratio <= 0:
            self._road_damaged.clear()
            # print(f"🚧 No road damage configured (ratio=0)")
            return
        
        # 获取所有激活的customer节点（排除depot和未激活的动态节点）
        delta = self.status_dict['delta']
        active_customers = []
        
        for i in range(self.num_nodes):
            # 跳过depot（index 0）
            if i == 0:
                continue
            # 只选择激活的节点（delta >= 0）
            if delta[i] >= 0:
                active_customers.append(i)
        
        if not active_customers:
            # print(f"🚧 No active customer nodes to damage")
            return
        
        # 计算需要损坏的节点数量
        num_damaged = max(1, int(len(active_customers) * road_damage_ratio))
        
        # 随机选择节点设置为道路损坏
        damaged_nodes = np.random.choice(active_customers, size=num_damaged, replace=False)
        self._road_damaged = set(damaged_nodes)
        
        # print(f"🚧 Road damage initialized: {num_damaged}/{len(active_customers)} nodes damaged")
        # print(f"   Damaged nodes: {sorted(self._road_damaged)}")
        
        # 添加到状态字典中供观测使用
        self.status_dict['road_damaged'] = np.zeros(self.num_nodes, dtype=int)
        for node in self._road_damaged:
            self.status_dict['road_damaged'][node] = 1

    def lookup_key(self, key, index=None):

        if key in set(self.status_dict.keys()):
            elem = self.status_dict[key]
        elif key in set(self.constants_dict.keys()):
            elem = self.constants_dict[key]
        elif key in set(self.signals_dict.keys()):
            elem = self.signals_dict[key]
        elif key in set(self.key_groups_dict.keys()):
            elem = self.key_groups_dict[key]
        else:
            raise KeyError('Key not found:', key)

        if not index is None:
            return elem[index]
        return elem


    def init_db(self):
        """创建各类内部字典，初始化全局时间与重要参数空间。"""
        # Clear previous
        self.status_dict.clear()
        self.constants_dict.clear()
        self.signals_dict.clear()

        # 各类索引列表
        self.d_indices = []
        self.c_indices = []
        self.v_indices = []

        # 全局时刻
        self.total_time = 0.0
        self.prev_total_time = 0.0
        self.current_time = 0.0

        # 【修改】将统计粒度拆细
        self.stats_truck_move_time = 0.0  # 卡车纯行驶时间
        self.stats_truck_wait_time = 0.0  # 卡车等待时间
        self.stats_drone_move_time = 0.0  # 无人机飞行时间 (含服务飞行+汇合飞行)
        self.stats_drone_wait_time = 0.0  # 无人机等待时间

        # 关键修复：为所有节点（包括动态节点）分配完整空间
        total_nodes = self.num_nodes  # 确保使用正确的总节点数
        
        # print(f"🔧 Initializing database with {total_nodes} total nodes")
        # print(f"   Static customers: {self.num_static_customers}")
        # print(f"   Dynamic customers: {self.num_dynamic_nodes}")
        # print(f"   Depots: {self.num_depots}")
        
        # 初始化所有节点的数组（包括未激活的动态节点）
        self.status_dict['delta'] = np.ones(total_nodes, dtype=int)
        self.status_dict['n_items'] = np.zeros(total_nodes, dtype=float)
        self.status_dict['n_coord'] = np.zeros((total_nodes, 2), dtype=float)
        
        # 初始化constants数组
        self.constants_dict['deadline'] = np.full(total_nodes, self.horizon, dtype=float)
        self.constants_dict['alpha'] = np.zeros(total_nodes, dtype=float)
        self.constants_dict['beta'] = np.zeros(total_nodes, dtype=float)
        self.constants_dict['n_type'] = np.zeros(total_nodes, dtype=int)
        
        # 关键修复：先设置动态节点为未激活状态（在创建静态节点之前）
        if self.dynamic_enabled and self.num_dynamic_nodes > 0:
            dynamic_start = self.num_depots + self.num_static_customers
            dynamic_end = self.num_nodes
            
            # print(f"🔧 Setting dynamic nodes {dynamic_start}-{dynamic_end-1} as inactive (delta=-1)")
            
            # 先将动态节点的delta设为-1
            for i in range(dynamic_start, dynamic_end):
                self.status_dict['delta'][i] = -1
            
            # print(f"✅ Initialized {self.num_dynamic_nodes} dynamic nodes as inactive (delta=-1)")
        
        # depot节点设置为已访问
        self.status_dict['delta'][0] = 0
        self.visited_nodes.add(0)

        # 初始化车辆状态
        self.status_dict['ET'] = np.zeros((self.num_trucks,), dtype=int)
        self.status_dict['LT'] = np.zeros((self.num_trucks,), dtype=float)
        self.status_dict['NT'] = np.zeros((self.num_trucks,), dtype=int)
        self.status_dict['TW'] = np.full((self.num_trucks,), self.WT_max, dtype=float)
        
        self.status_dict['ED'] = np.zeros((self.num_drones,), dtype=int)
        self.status_dict['LD'] = np.zeros((self.num_drones,), dtype=float)
        self.status_dict['ND'] = np.zeros((self.num_drones,), dtype=int)
        self.status_dict['DW'] = np.zeros((self.num_drones,), dtype=float)
        self.status_dict['attached_truck'] = np.zeros((self.num_drones,), dtype=int)
        self.status_dict['drone_battery'] = np.full((self.num_drones,), self.drone_battery, dtype=float)
        self.status_dict['drone_coord'] = np.zeros((self.num_drones,), dtype=int)
        
        # 兼容性：保持原有的车辆总数相关字段
        self.num_vehicles = self.num_trucks + self.num_drones

        # 无人机相关常量
        self.constants_dict['WD_max'] = np.full((self.num_drones,), self.WD_max, dtype=float)
        self.constants_dict['max_charge'] = np.full((self.num_drones,), self.drone_battery, dtype=float)

        # 车辆相关状态（兼容性保留）
        self.status_dict['v_dest'] = np.zeros((self.num_vehicles,), dtype=float)
        self.status_dict['v_to_n'] = np.zeros((self.num_vehicles,), dtype=int)
        self.status_dict['v_items'] = np.zeros((self.num_vehicles,), dtype=float)
        self.status_dict['v_weight'] = np.zeros((self.num_vehicles,), dtype=float)
        self.status_dict['v_coord'] = np.zeros((self.num_vehicles,), dtype=int)

        # 修复节点索引分配逻辑 - 只包含静态节点
        self.d_indices = list(range(self.num_depots))
        self.c_indices = list(range(self.num_depots, self.num_depots + self.num_static_customers))
        
        # 修复：显示正确的节点分配信息
        dynamic_indices = list(range(self.num_depots + self.num_static_customers, self.num_nodes))
        # print(f"🔧 Node indices allocation:")
        # print(f"  - depot_indices: {self.d_indices}")
        # print(f"  - static_customer_indices: {self.c_indices}")
        # print(f"  - reserved_dynamic_indices: {dynamic_indices}")
        # print(f"  - total nodes: {self.num_nodes}")

        # 设置节点类型常量
        for i in self.d_indices:
            self.constants_dict['n_type'][i] = 0  # depot
        for i in self.c_indices:
            self.constants_dict['n_type'][i] = 1  # customer
        # 动态节点已在上面设置

        # 填 min_max_dict
        self.min_max_dict = {
            'n_items': [10.0, 50.0],
            'deadline': [0.0, self.horizon],
            'TW': [0.0, self.WT_max],
            'DW': [0.0, self.WD_max],
            'LT_time': [0.0, self.horizon],
            'LD_time': [0.0, self.horizon],
            'time': [0.0, self.horizon],
        }

        # 路损受损节点集
        self._road_damaged = set()

        # 生成动态节点池（包含release time和属性）
        if self.dynamic_enabled and self.num_dynamic_nodes > 0:
            self.dynamic_nodes_pool.clear()
            dynamic_start = self.num_depots + self.num_static_customers

            # =====================================================================
            # 【开始修改区】：采用 Gu 等人的泊松过程生成动态出现时间 (Release Time)
            # =====================================================================
            
            # 1. 获取变异系数 lambda_var (若未配置则默认取 2.0)
            dyn_cfg = cfg['environment'].get('dynamic_nodes', {})
            lambda_var = float(dyn_cfg.get('lambda_var', 2.0))
            
            # 2. 按照论文设定，时间窗 delta 设置为 15 分钟
            delta_window = 15.0  
            num_periods = max(2, int(self.horizon / delta_window)) # 300 / 15 = 20 个周期
            
            # 3. 计算每个周期的期望到达订单数: λ * N_dynamic / (T_max/Δ - 1)
            expected_arrivals = lambda_var * self.num_dynamic_nodes / (num_periods - 1)
            
            gu_release_times = []
            unassigned_count = self.num_dynamic_nodes
            
            # 4. 从第 1 个周期开始，用泊松分布模拟每个周期内涌入的订单数
            for period in range(1, num_periods):
                if unassigned_count <= 0:
                    break
                
                # 采样当前 15 分钟窗口内的订单数量
                n_arrivals = np.random.poisson(expected_arrivals)
                n_arrivals = min(n_arrivals, unassigned_count)
                
                # 为该窗口内的每一笔订单，在15分钟内随机分配具体的精确出现时间（分钟）
                for _ in range(n_arrivals):
                    rt = (period * delta_window) + random.uniform(0, delta_window)
                    rt = min(rt, self.horizon - 1.0)  # 安全保护，确保不超过 horizon
                    gu_release_times.append(rt)
                    
                unassigned_count -= n_arrivals
                
            # 5. 如果因为泊松采样的波动，有剩余订单未分配完，统一兜底分配到最后的时间段
            for _ in range(unassigned_count):
                rt = (num_periods - 1) * delta_window + random.uniform(0, delta_window)
                rt = min(rt, self.horizon - 1.0)
                gu_release_times.append(rt)
                
            # 6. 打乱时间池，以保证后续空间坐标分配的随机性（避免先生成的节点总是在同一区域）
            random.shuffle(gu_release_times)
            # =====================================================================
            # 【结束修改区】
            # =====================================================================
            
            # 遍历为每一个动态节点附着属性
            for i in range(self.num_dynamic_nodes):
                node_idx = dynamic_start + i

                # 读取刚才用泊松过程生成的时间
                release_time = gu_release_times[i]
                
                # # 随机生成release time
                # # release_time = np.random.uniform(0, self.horizon * 0.8)
                # # release_frac = cfg['environment'].get('dynamic_nodes', {}).get('release_frac', 0.8)
                # # release_time = np.random.uniform(0, self.horizon * release_frac)
                # dyn = cfg['environment'].get('dynamic_nodes', {})
                # rel_min = float(dyn.get('release_min_frac', 0.5))
                # rel_max = float(dyn.get('release_max_frac', 0.9))
                # release_time = np.random.uniform(self.horizon*rel_min, self.horizon*rel_max)

                # 生成节点属性
                coord = random_coordinates(self.grid)
                demand = np.random.uniform(10, 50)
                # deadline = np.random.uniform(release_time + 50, self.horizon)
                # slack_lo, slack_hi = cfg['environment'].get('dynamic_nodes', {}).get('deadline_slack', [50, self.horizon])
                # # 防止上界超过 horizon
                # deadline_hi = min(self.horizon, release_time + slack_hi)
                # deadline_lo = min(deadline_hi, release_time + slack_lo)
                # deadline = np.random.uniform(deadline_lo, deadline_hi)
                # 调用统一的 Solomon 方法生成动态节点的 deadline
                deadline = self.generate_solomon_deadline(release_time=release_time, coord=coord)
                
                # 立即设置所有属性
                self.status_dict['n_coord'][node_idx] = coord
                self.status_dict['n_items'][node_idx] = demand
                self.constants_dict['deadline'][node_idx] = deadline
                self.constants_dict['n_type'][node_idx] = 1  # customer type
                self.constants_dict['alpha'][node_idx] = cfg['node']['customer']['alpha']
                self.constants_dict['beta'][node_idx] = cfg['node']['customer']['beta']
                
                # delta已经在前面设置为-1，这里不需要再设置
                
                # 记录到池中
                self.dynamic_nodes_pool.append({
                    'node_idx': node_idx,
                    'release_time': release_time
                })
                
                # print(f"Generated dynamic node {node_idx}: release_time={release_time:.1f}")
            
            self.dynamic_nodes_pool.sort(key=lambda x: x['release_time'])
            release_times = [f"{x['release_time']:.1f}" for x in self.dynamic_nodes_pool]
            # print(f"Generated {len(self.dynamic_nodes_pool)} dynamic nodes with release times: {release_times}")
            

        # unvisited 节点列表 - 只包含激活的节点
        self._unvisited = set()
        for i in range(self.num_nodes):
            if self.status_dict['delta'][i] == 1:  # 只包含激活且未访问的节点
                self._unvisited.add(i)

        self.initialize_road_damage()

        # 初始化车辆信号
        for sig in self.signal_list:
            self.signals_dict[sig] = np.zeros((self.num_vehicles,), dtype=float)
        
        # 最后验证初始化状态
        delta_counts = {
            -1: np.sum(self.status_dict['delta'] == -1),
            0: np.sum(self.status_dict['delta'] == 0),
            1: np.sum(self.status_dict['delta'] == 1)
        }
        # print(f"✅ Final delta state verification: inactive={delta_counts[-1]}, visited={delta_counts[0]}, active={delta_counts[1]}")
        # print(f"✅ Database initialization complete with {total_nodes} total nodes")

    def check_dynamic_nodes_activation(self):
        """检查是否有动态节点需要激活 - 修复时间推进问题"""
        if not self.dynamic_enabled or not self.dynamic_nodes_pool:
            return []
        
        # **关键修复：使用正确的当前时间**
        current_time = self.total_time  # 使用实际推进的时间
        
        # print(f"Checking dynamic nodes at time {current_time:.1f} (next check: {self.next_check_time:.1f})")
        
        # 只在检查时间点执行
        if current_time < self.next_check_time:
            return []
        
        # # 更新下次检查时间
        # self.next_check_time += self.delta_t
        
        # 方法2：对齐到delta_t的整数倍（更规整）
        import math
        next_check_multiplier = math.floor(current_time / self.delta_t) + 1
        self.next_check_time = next_check_multiplier * self.delta_t
        
        # print(f"Updated next_check_time to {self.next_check_time:.1f} (aligned to delta_t={self.delta_t})")
        
        activated_nodes = []
        nodes_to_remove = []
        
        for i, node in enumerate(self.dynamic_nodes_pool):
            if node['release_time'] <= current_time:
                # 激活此节点
                node_idx = self._activate_dynamic_node(node)
                activated_nodes.append(node_idx)
                nodes_to_remove.append(i)
                # print(f"Activated dynamic node {node_idx} at time {current_time:.1f} (release time: {node['release_time']:.1f})")
        
        # 从池中移除已激活的节点
        for i in reversed(nodes_to_remove):
            self.dynamic_nodes_pool.pop(i)
        
        return activated_nodes

    def _activate_dynamic_node(self, node_info):
        """激活一个动态节点 - 只需改变delta状态"""
        node_idx = node_info['node_idx']
        
        # 验证节点当前是未激活状态
        if self.status_dict['delta'][node_idx] != -1:
            # print(f"⚠️ Node {node_idx} is not inactive (delta={self.status_dict['delta'][node_idx]})")
            return node_idx
        
        # 只需要改变delta状态（属性已经在init_db中生成）
        self.status_dict['delta'][node_idx] = 1
        
        # 更新集合
        if node_idx not in self.c_indices:
            self.c_indices.append(node_idx)
        self.active_dynamic_nodes.add(node_idx)
        self._unvisited.add(node_idx)

        # 🚧 新增：按概率将动态节点设置为道路损坏
        env_cfg = cfg.get('environment', {})
        road_damage_ratio = env_cfg.get('road_damage_ratio')
        if road_damage_ratio > 0 and np.random.random() < road_damage_ratio:
            self._road_damaged.add(node_idx)
            if 'road_damaged' in self.status_dict:
                self.status_dict['road_damaged'][node_idx] = 1
            # print(f"   🚧 Dynamic node {node_idx} is ROAD DAMAGED!")
        
        # 读取预生成的属性用于日志
        coord = self.status_dict['n_coord'][node_idx]
        demand = self.status_dict['n_items'][node_idx]
        deadline = self.constants_dict['deadline'][node_idx]
        
        # print(f"✨ Activated node {node_idx} at time {self.total_time:.1f}")
        # print(f"   Properties: coord={coord}, demand={demand:.1f}, deadline={deadline:.1f}")
        # print(f"   Delta changed: -1 -> 1")
        # print(f"   Road damaged: {'Yes' if node_idx in self._road_damaged else 'No'}")
    
        return node_idx

    def get_current_node_count(self):
        """返回当前激活的节点总数"""
        # 添加安全检查
        if not hasattr(self, 'status_dict') or 'delta' not in self.status_dict:
            # print("Warning: Database not fully initialized, returning 0")
            return 0
        
        # 修复：正确计算激活节点数（delta >= 0的节点都是激活的）
        delta_array = self.status_dict['delta']
        active_count = np.sum(delta_array >= 0)
        
        # 添加调试信息
        # print(f"🔍 get_current_node_count debug:")
        # print(f"    Total nodes in array: {len(delta_array)}")
        # print(f"    Delta values: {delta_array}")
        # print(f"    Active nodes (delta>=0): {active_count}")
        # print(f"    Breakdown - visited(0): {np.sum(delta_array==0)}, active(1): {np.sum(delta_array==1)}, inactive(-1): {np.sum(delta_array==-1)}")
        
        return active_count

    def get_available_nodes_for_service(self):
        """返回当前可以被服务的节点（delta=1的节点）"""
        return [i for i, delta in enumerate(self.status_dict['delta']) if delta == 1]

    def is_node_active(self, node_idx):
        """检查节点是否已激活（可被访问）"""
        if node_idx < self.num_depots + self.num_static_customers:
            return True  # 静态节点总是激活的
        else:
            return self.status_dict['delta'][node_idx] != -1  # 动态节点检查delta值

    def add_node(self, node_obj, n_index, n_type):
        """添加节点 - 修复版本，防止覆盖动态节点的delta状态"""
        
        # 关键修复：检查是否为动态节点预留位置
        dynamic_start = self.num_depots + self.num_static_customers
        
        is_dynamic_reserved = (n_index >= dynamic_start)
        
        if is_dynamic_reserved:
            # print(f"⚠️ Skipping reserved dynamic position {n_index}")
            return  # 不允许在动态节点预留位置创建静态节点
        
        # 生成不重复的坐标（只与已激活节点比较）
        existing_coords = set()
        for i in range(self.num_nodes):
            if i < len(self.status_dict.get('n_coord', [])) and self.status_dict['delta'][i] >= 0:
                coord = tuple(self.status_dict['n_coord'][i])
                if coord != (0, 0):
                    existing_coords.add(coord)
        
        coord = random_coordinates(self.grid)
        while tuple(coord) in existing_coords:
            coord = random_coordinates(self.grid)
        
        # 设置坐标
        self.status_dict['n_coord'][n_index] = coord
        
        # 设置demand/stock
        if node_obj is not None:
            if hasattr(node_obj, 'init_demand'):
                init_val = node_obj.init_demand
            elif hasattr(node_obj, 'init_stock'):
                init_val = node_obj.init_stock
            else:
                init_val = 0
        else:
            if n_type == 1:  # customer
                init_val = np.random.uniform(10, 50)
            else:  # depot
                init_val = 0
        
        self.status_dict['n_items'][n_index] = init_val
        self.constants_dict['n_type'][n_index] = n_type
        
        # # 设置deadline
        # if n_type == 1:  # customer
        #     deadline_val = np.random.uniform(self.horizon * 0.2, self.horizon * 0.9)
        # else:  # depot
        #     deadline_val = self.horizon
        # 设置deadline
        if n_type == 1:  # customer
            # 静态节点在 t=0 时就已知，因此 release_time = 0.0
            deadline_val = self.generate_solomon_deadline(release_time=0.0, coord=coord)
        else:  # depot
            deadline_val = self.horizon
        
        self.constants_dict['deadline'][n_index] = deadline_val
        
        # 只为静态节点设置delta，不触碰动态节点
        if n_index == 0:  # depot
            self.status_dict['delta'][n_index] = 0  # 已访问
        elif n_index < dynamic_start:  # 静态customer
            self.status_dict['delta'][n_index] = 1  # 激活未访问
        # 不设置动态节点的delta，保持其-1状态
        
        # 更新索引列表
        if n_type == 0:  # depot
            if n_index not in self.d_indices:
                self.d_indices.append(n_index)
        else:  # customer
            if n_index not in self.c_indices:
                self.c_indices.append(n_index)
        
        # print(f"✅ Added node {n_index}: type={n_type}, demand={init_val:.2f}, "
            # f"deadline={deadline_val:.1f}, delta={self.status_dict['delta'][n_index]}")
        # print(f"✅ Added node {n_index}: type={n_type}, demand={init_val:.2f}")
        # # 在这里添加暂停，等待用户按回车继续
        # input("Press Enter to continue adding nodes...")

    # 修复depot_indices和customer_indices属性访问
    @property
    def depot_indices(self):
        return self.d_indices

    @property
    def customer_indices(self):
        return self.c_indices


    def add_vehicle(self, veh_obj, v_index, v_type):
        """
        添加一个车辆到库中，v_index 从 0 到 num_vehicles-1。
        veh_obj.is_truck 决定是否为卡车，veh_obj.v_loadable 表示是否可载货。
        """
        # 当前位置
        coord = random_coordinates(self.grid)
        insert_at_coord(self.status_dict, 'v_coord', coord, v_index, self.num_vehicles)
        # 是否空闲
        insert_at_array(self.status_dict, 'v_free', 1, v_index, self.num_vehicles)
        # 各种二值属性
        insert_at_array(self.constants_dict, 'v_is_truck', int(veh_obj.is_truck), v_index, self.num_vehicles)
        insert_at_array(self.constants_dict, 'v_loadable', int(veh_obj.v_loadable), v_index, self.num_vehicles)

        # 更新索引
        self.v_indices.append(v_index)
        self.key_groups_dict['vehicles'].append(f'v{v_index}')
        # print(">>> vehicles keys:", self.key_groups_dict['vehicles'])

    def get_val(self, key):
        """
        统一取值：
          - 'deadline','alpha','beta' → constants_dict
          - status_dict 中存在 → status_dict
          - 常量字典 → constants_dict
        """
        if key == 'deadline':
            # 在这里添加调试，看看deadline值什么时候被改变
            deadline_vals = self.constants_dict.get('deadline', np.array([]))
            # print(f"🔍 get_val('deadline') called, values: {deadline_vals}")
            return deadline_vals

        if key == 'demand':
            return self.status_dict['n_items']
        if key == 'n_items':
            return self.status_dict['n_items']
        if key in ('deadline', 'alpha', 'beta'):
            return self.constants_dict[key]
        if key == 'unassigned':
            return self.status_dict['delta']
        if key == 'road_damaged':  # 🚧 新增
            if 'road_damaged' in self.status_dict:
                return self.status_dict['road_damaged']
            else:
                # 返回全零数组表示没有道路损坏
                return np.zeros(self.num_nodes, dtype=int)
        if key == 'time':
            return np.array([self.current_time] * self.num_vehicles)
        # 新增：支持 DW 和 attached_truck
        if key == 'DW':
            return self.status_dict['DW']
        if key == 'attached_truck':
            return self.status_dict['attached_truck']
        if key in self.status_dict:
            return self.status_dict[key]
        if key in self.constants_dict:
            return self.constants_dict[key]
        raise KeyError(f"Key {key} not found in temp_db")

    def generate_solomon_deadline(self, release_time, coord):
        """
        严格按照 Solomon (1987) 方法生成 Deadline (适用于静态和动态节点)
        """
        # 1. 获取 Depot(索引0) 的坐标
        depot_coord = self.status_dict['n_coord'][0]
        
        # 2. 计算曼哈顿网格距离并转为物理距离
        grid_distance = float(np.sum(np.abs(np.array(coord) - depot_coord)))
        dist_to_depot = grid_distance * (self.area_size / self.grid[0])
        
        # 3. 计算卡车单程物理行驶时间
        travel_time_depot = dist_to_depot / self.truck_speed
        
        # 4. 计算 Solomon 区间的上下限
        # earliest_arrival: 最早可能到达的时间 = 释放时间 + 行驶时间
        earliest_arrival = release_time + travel_time_depot
        # latest_start: 最晚必须开始服务的时间 = 总周期 - 回程时间
        latest_start = self.horizon - travel_time_depot
        
        # 逻辑保护：如果节点太远，保证区间不倒置
        if latest_start < earliest_arrival:
            latest_start = earliest_arrival + 5.0
            
        # 5. 生成时间窗中心点 Center ~ U(earliest_arrival, latest_start)
        center_time = np.random.uniform(earliest_arrival, latest_start)
        
        # 6. 生成时间窗半宽 Half-width ~ N(μ, σ^2)
        # 读取 cfg 中的 slack 参数并映射为正态分布的均值和标准差，保留调参能力
        slack_lo, slack_hi = cfg['environment'].get('dynamic_nodes', {}).get('deadline_slack', [5, 30])
        mean_slack = (slack_lo + slack_hi) / 2.0
        std_slack = max((slack_hi - slack_lo) / 4.0, 1.0)
        
        half_width = np.abs(np.random.normal(loc=mean_slack, scale=std_slack))
        
        # 7. 计算最终 Deadline 并确保不超过 horizon
        deadline = min(center_time + half_width, self.horizon)
        return deadline
    
    def get_available_truck_indices(self):
        """获取可决策的卡车列表"""
        return list(range(self.num_trucks))
    
    def get_available_drone_indices(self):
        """获取可决策的无人机列表"""
        return list(range(self.num_drones))

    def update_drone_attachment(self, drone_idx, truck_idx):
        """更新无人机附属关系"""
        if 0 <= drone_idx < self.num_drones and 0 <= truck_idx < self.num_trucks:
            self.status_dict['attached_truck'][drone_idx] = truck_idx
            # 更新无人机位置为附属卡车的位置
            if drone_idx + self.num_trucks < len(self.status_dict['v_coord']):
                self.status_dict['v_coord'][drone_idx + self.num_trucks] = self.status_dict['v_coord'][truck_idx]

    def get_unvisited_nodes(self):
        """返回当前 Ct = CNt∪CUt，即所有还没被访问过的节点索引列表。"""
        return list(self._unvisited)

    def mark_visited(self, node_idx):
        """在节点 i 被访问后调用，移出 unvisited。"""
        """
        调用时标记节点已访问：
         - 从未访问集合中删除
         - status_dict['delta'] 置 0
         - visited_nodes 增加该节点
        """
        self._unvisited.discard(node_idx)
        # 同时将 delta 置 0（已指派/已访问）
        self.status_dict['delta'][node_idx] = 0
        self.visited_nodes.add(node_idx)

    def get_unvisited_nodes(self):
        return list(self._unvisited)

    def get_road_damaged_nodes(self):
        """返回 RDt：道路受损、卡车无法通行的节点索引列表。"""
        return list(self._road_damaged)

    def set_road_damaged(self, node_idx_list):
        """设置哪些节点暂时不能被卡车访问。"""
        self._road_damaged = set(node_idx_list)

    def drone_can_service_and_return(self, k, node_idx):
        """
        判断无人机 k 是否有足够电量/载重去访问 node_idx 并返航到卡车位置。
        这里示例：用 WD_max 直接比较需求，实际还要考虑距离、电量等。
        """
        demand = self.get_val('n_items')[node_idx]
        return demand <= self.get_val('WD_max')[k]

    def get_available_tandem_indices(self):
        """
        返回可决策的 tandem(k) 列表 U_t。
        此处简化：直接返回所有车辆索引，你可根据 ET/ED 做更精细过滤。
        """
        return list(self.v_indices)

    # def distance(self, src, dst):
    #     """
    #     计算从 src 到 dst 的欧氏距离：
    #     - src: 若是 int，视为节点索引，坐标从 status_dict['n_coord'] 读
    #             否则如果是 tuple/list/ndarray，直接当坐标
    #     - dst: 节点索引，坐标从 status_dict['n_coord'] 读
    #     """
    #     # 获取源坐标
    #     if isinstance(src, (int, np.integer)):
    #         # src是节点索引，从n_coord获取坐标
    #         coord_src = self.status_dict['n_coord'][src]
    #     else:
    #         # src是坐标数组
    #         coord_src = np.array(src, dtype=float)
        
    #     # 获取目标坐标
    #     coord_dst = self.status_dict['n_coord'][dst]
        
    #     # # 计算欧氏距离
    #     # distance_val = float(np.linalg.norm(coord_src - coord_dst))
        
        
    #     # 计算网格距离
    #     grid_distance = float(np.linalg.norm(coord_src - coord_dst))
        
    #     # 转换为实际公里数
    #     grid_size = self.area_size / self.grid[0]  # 每个格子的实际大小（km）
    #     distance_val = grid_distance * grid_size

    #     # 调试输出
    #     # print(f"     📐 Distance calculation: src={src}->coord{coord_src}, dst={dst}->coord{coord_dst}, distance={distance_val:.3f}")
        
        
    #     return distance_val
    def distance(self, src, dst, metric='euclidean'):
        """
        计算两点之间的物理距离
        :param src: 起点索引或坐标
        :param dst: 终点索引
        :param metric: 距离度量方式，'euclidean' (无人机) 或 'manhattan' (卡车)
        :return: 实际距离（km）
        """
        if isinstance(src, (int, np.integer)):
            # src是节点索引，从n_coord获取坐标
            coord_src = self.status_dict['n_coord'][src]
        else:
            # src是坐标数组
            coord_src = np.array(src, dtype=float)
        
        # 获取目标坐标
        coord_dst = self.status_dict['n_coord'][dst]
        
        # 根据 metric 计算不同的网格距离
        if metric == 'manhattan':
            # 曼哈顿距离：|x1-x2| + |y1-y2| (卡车)
            grid_distance = float(np.sum(np.abs(coord_src - coord_dst)))
        else:
            # 欧式距离：sqrt((x1-x2)^2 + (y1-y2)^2) (无人机，默认)
            grid_distance = float(np.linalg.norm(coord_src - coord_dst))
        
        # 转换为实际公里数
        grid_size = self.area_size / self.grid[0]  # 每个格子的实际大小（km）
        distance_val = grid_distance * grid_size
        
        return distance_val


    def total_time_delta(self):
        """返回本步时间增量，并更新 prev_total_time。"""
        delta = self.total_time - self.prev_total_time
        self.prev_total_time = self.total_time
        return delta

    def terminal_state(self):
        """
        判断终止条件：所有客户需求为 0 且车辆都回到任意一个 depot 上。
        """
        # 所有客户需求清零
        if np.all(self.status_dict['n_items'][self.c_indices] == 0):
            # 检查每辆车是否在某个 depot 坐标上
            dep_coords = self.status_dict['n_coord'][self.d_indices]
            for node_idx in self.status_dict['truck_node']:
                if node_idx not in self.depot_indices:
                    return False
            return True
        return False
