import json
import math
import copy
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from trucks_and_drones.config import cfg

ENV_CFG = cfg['environment']
NODE_CFG = cfg['node']

NUM_TRUCKS = ENV_CFG.get('num_trucks', 5)                
NUM_DRONES = ENV_CFG.get('num_drones', 5)                
TRUCK_SPEED = ENV_CFG.get('truck_speed', 0.5)            
DRONE_SPEED = ENV_CFG.get('drone_speed', 1.0)            
MAX_WORKING_HOURS = ENV_CFG.get('horizon', 300.0)        
DRONE_MAX_ENDURANCE = ENV_CFG.get('max_charge', 50.0)    
DRONE_MAX_CAPACITY = ENV_CFG.get('WD_max', 80.0)         
TRUCK_COST = ENV_CFG.get('ct_cost', 0.083)               
DRONE_COST = ENV_CFG.get('cd_cost', 0.021)               

UNIT_REWARD_P = NODE_CFG['customer'].get('beta', 30.0)         
UNSERVED_PENALTY = NODE_CFG['customer'].get('beta', 30.0)      
DELAY_ALPHA = NODE_CFG['customer'].get('alpha', 0.5)           

COST_SCALE = 1000  

def calc_manhattan_dist(n1, n2):
    """卡车走城市路网：曼哈顿距离"""
    return abs(n1['x'] - n2['x']) + abs(n1['y'] - n2['y'])

def calc_euclidean_dist(n1, n2):
    """无人机飞直线：欧式距离"""
    return math.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])

class RouteSegment:
    def __init__(self, start_node, end_node):
        self.start_node = start_node
        self.end_node = end_node
        self.truck_nodes = []  
        self.drone_nodes = []  
        self._update_stats()

    def _update_stats(self):
        t_dist = 0.0
        curr = self.start_node
        for n in self.truck_nodes:
            t_dist += calc_manhattan_dist(curr, n)
            curr = n
        t_dist += calc_manhattan_dist(curr, self.end_node)
        self.dur_ts = t_dist / TRUCK_SPEED

        d_dist = 0.0
        if not self.drone_nodes:
            self.dur_dt = 0.0
        else:
            curr = self.start_node
            for n in self.drone_nodes:
                d_dist += calc_euclidean_dist(curr, n)
                curr = n
            d_dist += calc_euclidean_dist(curr, self.end_node)
            self.dur_dt = d_dist / DRONE_SPEED
            
        self.duration = max(self.dur_ts, self.dur_dt)
        self.cost_ts = self.dur_ts * TRUCK_COST
        self.cost_dt = self.dur_dt * DRONE_COST
        
        self.drone_energy = self.duration if self.drone_nodes else 0.0
        self.drone_load = sum(n.get('demand', 1) for n in self.drone_nodes)


class GuDynamicSolver:
    def __init__(self, dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.nodes = {n['id']: n for n in data['nodes']}
        self.depot = self.nodes[0]
        
        self.static_truck_customers = [
            n for n in data['nodes'] 
            if not n['is_dynamic'] and n['id'] != 0 and n.get('road_damaged', 0) == 0
        ]
        
        self.pending_customers = [
            n for n in data['nodes'] 
            if (n['is_dynamic'] or n.get('road_damaged', 0) == 1) and n['id'] != 0
        ]
        self.pending_customers.sort(key=lambda x: x['release_time'])
        
        self.fleet_segments = [ [] for _ in range(NUM_TRUCKS) ]
        self.served_count = 0

    def solve_pdptw_with_ortools(self, nodes_to_visit):
        routing_nodes = [self.depot] + nodes_to_visit
        num_nodes = len(routing_nodes)
        
        manager = pywrapcp.RoutingIndexManager(num_nodes, NUM_TRUCKS, 0)
        routing = pywrapcp.RoutingModel(manager)

        def cost_callback(from_index, to_index):
            if from_index == to_index: return 0
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travel_time = calc_manhattan_dist(routing_nodes[from_node], routing_nodes[to_node]) / TRUCK_SPEED
            return int(travel_time * TRUCK_COST * COST_SCALE)

        cost_callback_index = routing.RegisterTransitCallback(cost_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(cost_callback_index)

        def time_callback(from_index, to_index):
            if from_index == to_index: return 0
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travel_time = calc_manhattan_dist(routing_nodes[from_node], routing_nodes[to_node]) / TRUCK_SPEED
            return int(travel_time * 100)

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            30000, int(MAX_WORKING_HOURS * 100), False, 'Time')
        time_dimension = routing.GetDimensionOrDie('Time')

        # 增加卡车载重约束维度 (保证初始静态路线合法)
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(routing_nodes[from_node].get('demand', 0))

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  
            [int(ENV_CFG.get('WT_max', 1000.0))] * NUM_TRUCKS, 
            True, 
            'Capacity')
        
        for i, n in enumerate(routing_nodes):
            if i == 0: continue
            index = manager.NodeToIndex(i)
            start = int(n.get('release_time', 0) * 100)
            deadline = int(n.get('deadline', MAX_WORKING_HOURS) * 100)
            max_time = int(MAX_WORKING_HOURS * 100)
            time_dimension.CumulVar(index).SetRange(start, max_time)
            penalty_rate = int((DELAY_ALPHA * COST_SCALE) / 100)
            time_dimension.SetCumulVarSoftUpperBound(index, deadline, penalty_rate)

        real_drop_penalty = int((UNIT_REWARD_P + UNSERVED_PENALTY) * COST_SCALE)
        for node in range(1, num_nodes):
            routing.AddDisjunction([manager.NodeToIndex(node)], real_drop_penalty)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 10

        solution = routing.SolveWithParameters(search_parameters)
        if not solution: raise ValueError("OR-Tools 无解！")

        all_routes = []
        for vehicle_id in range(NUM_TRUCKS):
            index = routing.Start(vehicle_id)
            route_sequence = []
            while not routing.IsEnd(index):
                route_sequence.append(routing_nodes[manager.IndexToNode(index)])
                index = solution.Value(routing.NextVar(index))
            route_sequence.append(routing_nodes[manager.IndexToNode(index)]) 
            if len(route_sequence) > 2: all_routes.append(route_sequence)
        return all_routes

    def reassignment_procedure(self, all_truck_routes):
        self.fleet_segments = [ [] for _ in range(NUM_TRUCKS) ]
        for truck_idx, truck_route in enumerate(all_truck_routes):
            curr_launch_node = truck_route[0]
            i = 1
            while i < len(truck_route) - 1:
                candidate_drone_nodes, retrieval_node, best_valid_drone_nodes = [], None, []
                for j in range(i, len(truck_route) - 1):
                    test_node = truck_route[j]
                    candidate_drone_nodes.append(test_node)
                    test_retrieval = truck_route[j + 1]
                    
                    if test_retrieval.get('road_damaged', 0) == 1:
                        break
                        
                    temp_seg = RouteSegment(curr_launch_node, test_retrieval)
                    temp_seg.drone_nodes = list(candidate_drone_nodes)
                    temp_seg.truck_nodes = [] 
                    temp_seg._update_stats()
                    
                    if temp_seg.drone_load <= DRONE_MAX_CAPACITY and temp_seg.drone_energy <= DRONE_MAX_ENDURANCE:
                        retrieval_node = test_retrieval
                        best_valid_drone_nodes = list(candidate_drone_nodes)
                    else: break 
                        
                if retrieval_node:
                    seg = RouteSegment(curr_launch_node, retrieval_node)
                    seg.drone_nodes = best_valid_drone_nodes
                    self.fleet_segments[truck_idx].append(seg)
                    self.served_count += len(best_valid_drone_nodes)
                    if retrieval_node['id'] != self.depot['id']: self.served_count += 1
                    curr_launch_node, i = retrieval_node, i + len(best_valid_drone_nodes) + 1
                else:
                    seg = RouteSegment(curr_launch_node, truck_route[i])
                    seg.truck_nodes = []
                    self.fleet_segments[truck_idx].append(seg)
                    if truck_route[i].get('road_damaged', 0) == 0 and truck_route[i]['id'] != self.depot['id']:
                        self.served_count += 1
                    curr_launch_node, i = truck_route[i], i + 1
                    
            if curr_launch_node['id'] != self.depot['id']:
                self.fleet_segments[truck_idx].append(RouteSegment(curr_launch_node, self.depot))

    def calculate_route_real_cost(self, segments):
        travel_cost = 0.0
        delay_penalty = 0.0
        curr_time = 0.0
        for seg in segments:
            curr_time += seg.duration
            travel_cost += (seg.dur_ts * TRUCK_COST) + (seg.dur_dt * DRONE_COST)
            
            sub_time_t = curr_time - seg.duration
            curr_pos_t = seg.start_node
            for tn in seg.truck_nodes:
                sub_time_t += calc_manhattan_dist(curr_pos_t, tn) / TRUCK_SPEED
                deadline = tn.get('deadline', MAX_WORKING_HOURS)
                if sub_time_t > deadline: delay_penalty += (sub_time_t - deadline) * DELAY_ALPHA
                curr_pos_t = tn
                
            sub_time_d = curr_time - seg.duration
            curr_pos_d = seg.start_node
            for dn in seg.drone_nodes:
                sub_time_d += calc_euclidean_dist(curr_pos_d, dn) / DRONE_SPEED
                deadline = dn.get('deadline', MAX_WORKING_HOURS)
                if sub_time_d > deadline: delay_penalty += (sub_time_d - deadline) * DELAY_ALPHA
                curr_pos_d = dn
                
            end_deadline = seg.end_node.get('deadline', MAX_WORKING_HOURS)
            if curr_time > end_deadline:
                delay_penalty += (curr_time - end_deadline) * DELAY_ALPHA
                
            if curr_time > MAX_WORKING_HOURS:
                delay_penalty += (curr_time - MAX_WORKING_HOURS) * DELAY_ALPHA 
                
        return travel_cost, delay_penalty

    # =========================================================================
    # 🌟 核心修改区：加入真实的载重重置逻辑与回仓补给（Multi-Trip）启发式
    # =========================================================================

    def validate_route_capacity(self, segments):
        """
        按时间顺序遍历整条线路，模拟卡车行驶中的货物消耗。
        一旦遇到仓库节点(id == 0)，则代表完成补给，当前载重清零。
        """
        trip_load = 0.0
        WT_max = ENV_CFG.get('WT_max', 1000.0)
        
        for seg in segments:
            # 起点若是仓库，清空载重
            if seg.start_node['id'] == 0:
                trip_load = 0.0
            
            # 累加卡车路径节点消耗
            for n in seg.truck_nodes:
                if n['id'] == 0:
                    trip_load = 0.0
                else:
                    trip_load += n.get('demand', 0)
                if trip_load > WT_max: 
                    return False
            
            # 累加无人机路径节点消耗（无人机的货是从卡车上带走的）
            for n in seg.drone_nodes:
                if n['id'] != 0:
                    trip_load += n.get('demand', 0)
                if trip_load > WT_max: 
                    return False
            
            # 终点若是仓库，清空载重
            if seg.end_node['id'] == 0:
                trip_load = 0.0
                
        return True

    def try_insert_truck(self, test_segments, s_idx, new_node, insert_idx):
        """常规直接插入策略"""
        seg = test_segments[s_idx]
        if new_node.get('road_damaged', 0) == 1: return False
        
        seg.truck_nodes.insert(insert_idx, new_node)
        seg._update_stats()
        
        if seg.duration > DRONE_MAX_ENDURANCE and seg.drone_nodes: return False
        if not self.validate_route_capacity(test_segments): return False
        return True

    def try_insert_truck_with_depot(self, test_segments, s_idx, new_node, insert_idx):
        """
        🚀 启发式主动回仓补货策略 (Multi-trip Heuristic)：
        如果没货了，卡车尝试在送货前强制塞入一个 Node 0 行程，进行补血。
        """
        seg = test_segments[s_idx]
        if new_node.get('road_damaged', 0) == 1: return False
        
        # 强制塞入仓库节点，再塞入新客户节点
        seg.truck_nodes.insert(insert_idx, self.depot)
        seg.truck_nodes.insert(insert_idx + 1, new_node)
        seg._update_stats()
        
        if seg.duration > DRONE_MAX_ENDURANCE and seg.drone_nodes: return False
        if not self.validate_route_capacity(test_segments): return False
        return True

    def try_insert_drone(self, test_segments, s_idx, new_node, insert_idx):
        """无人机插入策略"""
        seg = test_segments[s_idx]
        
        # 无人机单次起降的独立载重限制
        if seg.drone_load + new_node.get('demand', 0) > DRONE_MAX_CAPACITY: return False
        
        seg.drone_nodes.insert(insert_idx, new_node)
        seg._update_stats()
        
        if seg.duration > DRONE_MAX_ENDURANCE: return False
        if not self.validate_route_capacity(test_segments): return False
        return True

    def online_dynamic_insertion(self):
        accepted, rejected = 0, 0
        for dyn_node in self.pending_customers:
            best_delta_cost = float('inf')
            best_truck_idx, best_seg_idx, best_action = -1, -1, None
            
            for t_idx, segments in enumerate(self.fleet_segments):
                base_travel, base_delay = self.calculate_route_real_cost(segments)
                base_cost = base_travel + base_delay
                
                for s_idx, seg in enumerate(segments):
                    # =============== 尝试放入卡车 ===============
                    for i in range(len(seg.truck_nodes) + 1):
                        # 方案 A：常规插入
                        test_segments = copy.deepcopy(segments)
                        if self.try_insert_truck(test_segments, s_idx, dyn_node, i):
                            new_t, new_d = self.calculate_route_real_cost(test_segments)
                            new_cost = new_t + new_d
                            if new_cost - base_cost < best_delta_cost:
                                best_delta_cost = new_cost - base_cost
                                best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'truck', 'idx':i, 'with_depot': False}
                        
                        # 方案 B：顺路回仓补货后插入 (突破单次运力限制)
                        test_segments_depot = copy.deepcopy(segments)
                        if self.try_insert_truck_with_depot(test_segments_depot, s_idx, dyn_node, i):
                            new_t, new_d = self.calculate_route_real_cost(test_segments_depot)
                            new_cost = new_t + new_d
                            if new_cost - base_cost < best_delta_cost:
                                best_delta_cost = new_cost - base_cost
                                best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'truck', 'idx':i, 'with_depot': True}
                                
                    # =============== 尝试放入无人机 ===============
                    for i in range(len(seg.drone_nodes) + 1):
                        test_segments = copy.deepcopy(segments)
                        if self.try_insert_drone(test_segments, s_idx, dyn_node, i):
                            new_t, new_d = self.calculate_route_real_cost(test_segments)
                            new_cost = new_t + new_d
                            if new_cost - base_cost < best_delta_cost:
                                best_delta_cost = new_cost - base_cost
                                best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'drone', 'idx':i, 'with_depot': False}

            # =============== 确认并执行接单逻辑 ===============
            if best_truck_idx != -1 and best_delta_cost < (UNIT_REWARD_P + UNSERVED_PENALTY):
                test_segments = copy.deepcopy(self.fleet_segments[best_truck_idx])
                
                if best_action['type'] == 'truck':
                    if best_action.get('with_depot'):
                        self.try_insert_truck_with_depot(test_segments, best_seg_idx, dyn_node, best_action['idx'])
                    else:
                        self.try_insert_truck(test_segments, best_seg_idx, dyn_node, best_action['idx'])
                else:
                    self.try_insert_drone(test_segments, best_seg_idx, dyn_node, best_action['idx'])
                
                self.fleet_segments[best_truck_idx] = test_segments
                self.served_count += 1
                accepted += 1
            else:
                rejected += 1

    # =========================================================================

    def calculate_final_profit(self):
        travel_cost = 0.0
        delay_penalty = 0.0
        makespan = 0.0
        
        for segments in self.fleet_segments:
            t_cost, d_pen = self.calculate_route_real_cost(segments)
            travel_cost += t_cost
            delay_penalty += d_pen
            makespan = max(makespan, sum(seg.duration for seg in segments))
            
        service_reward_sum = self.served_count * UNIT_REWARD_P
        total_customers = len(self.nodes) - 1 
        unserved_count = total_customers - self.served_count
        unserved_penalty_sum = unserved_count * UNSERVED_PENALTY
        
        real_reward = service_reward_sum - travel_cost - delay_penalty - unserved_penalty_sum
        
        return {
            'served_count': self.served_count,
            'unserved_count': unserved_count,
            'service_reward_sum': service_reward_sum,
            'travel_cost': travel_cost,
            'delay_penalty': delay_penalty,
            'unserved_penalty': unserved_penalty_sum,
            'real_reward': real_reward,
            'time_used': makespan
        }

    def print_detailed_schedule(self):
        pass

    def solve(self):
        all_truck_routes = self.solve_pdptw_with_ortools(self.static_truck_customers)
        self.reassignment_procedure(all_truck_routes)
        self.online_dynamic_insertion()
        stats = self.calculate_final_profit()
        return stats

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dynamic_instance.json")
    args = parser.parse_args()
    if os.path.exists(args.dataset):
        GuDynamicSolver(args.dataset).solve()

# import json
# import math
# import copy
# from ortools.constraint_solver import routing_enums_pb2
# from ortools.constraint_solver import pywrapcp

# from trucks_and_drones.config import cfg

# ENV_CFG = cfg['environment']
# NODE_CFG = cfg['node']

# NUM_TRUCKS = ENV_CFG.get('num_trucks', 5)                
# NUM_DRONES = ENV_CFG.get('num_drones', 5)                
# TRUCK_SPEED = ENV_CFG.get('truck_speed', 0.5)            
# DRONE_SPEED = ENV_CFG.get('drone_speed', 1.0)            
# MAX_WORKING_HOURS = ENV_CFG.get('horizon', 300.0)        
# DRONE_MAX_ENDURANCE = ENV_CFG.get('max_charge', 50.0)    
# DRONE_MAX_CAPACITY = ENV_CFG.get('WD_max', 80.0)         
# TRUCK_COST = ENV_CFG.get('ct_cost', 0.083)               
# DRONE_COST = ENV_CFG.get('cd_cost', 0.021)               

# UNIT_REWARD_P = NODE_CFG['customer'].get('beta', 30.0)         
# UNSERVED_PENALTY = NODE_CFG['customer'].get('beta', 30.0)      
# DELAY_ALPHA = NODE_CFG['customer'].get('alpha', 0.5)           

# COST_SCALE = 1000  

# # ==========================================
# # 🌟 核心修复 1：严格对齐 RL 的距离测算口径
# # ==========================================
# def calc_manhattan_dist(n1, n2):
#     """卡车走城市路网：曼哈顿距离"""
#     return abs(n1['x'] - n2['x']) + abs(n1['y'] - n2['y'])

# def calc_euclidean_dist(n1, n2):
#     """无人机飞直线：欧式距离"""
#     return math.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])

# class RouteSegment:
#     def __init__(self, start_node, end_node):
#         self.start_node = start_node
#         self.end_node = end_node
#         self.truck_nodes = []  
#         self.drone_nodes = []  
#         self._update_stats()

#     def _update_stats(self):
#         # 卡车采用曼哈顿距离计算
#         t_dist = 0.0
#         curr = self.start_node
#         for n in self.truck_nodes:
#             t_dist += calc_manhattan_dist(curr, n)
#             curr = n
#         t_dist += calc_manhattan_dist(curr, self.end_node)
#         self.dur_ts = t_dist / TRUCK_SPEED

#         # 无人机采用欧式距离计算
#         d_dist = 0.0
#         if not self.drone_nodes:
#             self.dur_dt = 0.0
#         else:
#             curr = self.start_node
#             for n in self.drone_nodes:
#                 d_dist += calc_euclidean_dist(curr, n)
#                 curr = n
#             d_dist += calc_euclidean_dist(curr, self.end_node)
#             self.dur_dt = d_dist / DRONE_SPEED
            
#         self.duration = max(self.dur_ts, self.dur_dt)
#         self.cost_ts = self.dur_ts * TRUCK_COST
#         self.cost_dt = self.dur_dt * DRONE_COST
        
#         self.drone_energy = self.duration if self.drone_nodes else 0.0
#         self.drone_load = sum(n.get('demand', 1) for n in self.drone_nodes)


# class GuDynamicSolver:
#     def __init__(self, dataset_path):
#         with open(dataset_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         self.nodes = {n['id']: n for n in data['nodes']}
#         self.depot = self.nodes[0]
        
#         self.static_truck_customers = [
#             n for n in data['nodes'] 
#             if not n['is_dynamic'] and n['id'] != 0 and n.get('road_damaged', 0) == 0
#         ]
        
#         self.pending_customers = [
#             n for n in data['nodes'] 
#             if (n['is_dynamic'] or n.get('road_damaged', 0) == 1) and n['id'] != 0
#         ]
#         self.pending_customers.sort(key=lambda x: x['release_time'])
        
#         self.fleet_segments = [ [] for _ in range(NUM_TRUCKS) ]
#         self.served_count = 0

#     def solve_pdptw_with_ortools(self, nodes_to_visit):
#         routing_nodes = [self.depot] + nodes_to_visit
#         num_nodes = len(routing_nodes)
        
#         manager = pywrapcp.RoutingIndexManager(num_nodes, NUM_TRUCKS, 0)
#         routing = pywrapcp.RoutingModel(manager)

#         def cost_callback(from_index, to_index):
#             if from_index == to_index: return 0
#             from_node = manager.IndexToNode(from_index)
#             to_node = manager.IndexToNode(to_index)
#             # OR-Tools 底层规划卡车路线，必须用曼哈顿距离
#             travel_time = calc_manhattan_dist(routing_nodes[from_node], routing_nodes[to_node]) / TRUCK_SPEED
#             return int(travel_time * TRUCK_COST * COST_SCALE)

#         cost_callback_index = routing.RegisterTransitCallback(cost_callback)
#         routing.SetArcCostEvaluatorOfAllVehicles(cost_callback_index)

#         def time_callback(from_index, to_index):
#             if from_index == to_index: return 0
#             from_node = manager.IndexToNode(from_index)
#             to_node = manager.IndexToNode(to_index)
#             travel_time = calc_manhattan_dist(routing_nodes[from_node], routing_nodes[to_node]) / TRUCK_SPEED
#             return int(travel_time * 100)

#         time_callback_index = routing.RegisterTransitCallback(time_callback)
#         routing.AddDimension(
#             time_callback_index,
#             30000, int(MAX_WORKING_HOURS * 100), False, 'Time')
#         time_dimension = routing.GetDimensionOrDie('Time')

#         # 增加卡车载重约束维度
#         def demand_callback(from_index):
#             from_node = manager.IndexToNode(from_index)
#             return int(routing_nodes[from_node].get('demand', 0))

#         demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
#         routing.AddDimensionWithVehicleCapacity(
#             demand_callback_index,
#             0,  # capacity slack
#             [int(ENV_CFG.get('WT_max', 1000.0))] * NUM_TRUCKS,  # 每辆卡车的最大载重
#             True,  # start cumul to zero
#             'Capacity')
        
#         for i, n in enumerate(routing_nodes):
#             if i == 0: continue
#             index = manager.NodeToIndex(i)
#             start = int(n.get('release_time', 0) * 100)
#             deadline = int(n.get('deadline', MAX_WORKING_HOURS) * 100)
#             max_time = int(MAX_WORKING_HOURS * 100)
#             time_dimension.CumulVar(index).SetRange(start, max_time)
#             penalty_rate = int((DELAY_ALPHA * COST_SCALE) / 100)
#             time_dimension.SetCumulVarSoftUpperBound(index, deadline, penalty_rate)

#         real_drop_penalty = int((UNIT_REWARD_P + UNSERVED_PENALTY) * COST_SCALE)
#         for node in range(1, num_nodes):
#             routing.AddDisjunction([manager.NodeToIndex(node)], real_drop_penalty)

#         search_parameters = pywrapcp.DefaultRoutingSearchParameters()
#         search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
#         search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
#         search_parameters.time_limit.seconds = 15

#         # # 🌟 动态时间上限：根据节点规模给足思考时间
#         # if len(self.nodes) > 60:
#         #     search_parameters.time_limit.seconds = 20
#         # elif len(self.nodes) > 30:
#         #     search_parameters.time_limit.seconds = 10
#         # else:
#         #     search_parameters.time_limit.seconds = 5

#         solution = routing.SolveWithParameters(search_parameters)
#         if not solution: raise ValueError("OR-Tools 无解！")

#         all_routes = []
#         for vehicle_id in range(NUM_TRUCKS):
#             index = routing.Start(vehicle_id)
#             route_sequence = []
#             while not routing.IsEnd(index):
#                 route_sequence.append(routing_nodes[manager.IndexToNode(index)])
#                 index = solution.Value(routing.NextVar(index))
#             route_sequence.append(routing_nodes[manager.IndexToNode(index)]) 
#             if len(route_sequence) > 2: all_routes.append(route_sequence)
#         return all_routes

#     def reassignment_procedure(self, all_truck_routes):
#         self.fleet_segments = [ [] for _ in range(NUM_TRUCKS) ]
#         for truck_idx, truck_route in enumerate(all_truck_routes):
#             curr_launch_node = truck_route[0]
#             i = 1
#             while i < len(truck_route) - 1:
#                 candidate_drone_nodes, retrieval_node, best_valid_drone_nodes = [], None, []
#                 for j in range(i, len(truck_route) - 1):
#                     test_node = truck_route[j]
#                     candidate_drone_nodes.append(test_node)
#                     test_retrieval = truck_route[j + 1]
                    
#                     if test_retrieval.get('road_damaged', 0) == 1:
#                         break
                        
#                     temp_seg = RouteSegment(curr_launch_node, test_retrieval)
#                     temp_seg.drone_nodes = list(candidate_drone_nodes)
#                     temp_seg.truck_nodes = [] 
#                     temp_seg._update_stats()
                    
#                     if temp_seg.drone_load <= DRONE_MAX_CAPACITY and temp_seg.drone_energy <= DRONE_MAX_ENDURANCE:
#                         retrieval_node = test_retrieval
#                         best_valid_drone_nodes = list(candidate_drone_nodes)
#                     else: break 
                        
#                 if retrieval_node:
#                     seg = RouteSegment(curr_launch_node, retrieval_node)
#                     seg.drone_nodes = best_valid_drone_nodes
#                     self.fleet_segments[truck_idx].append(seg)
#                     self.served_count += len(best_valid_drone_nodes)
#                     if retrieval_node['id'] != self.depot['id']: self.served_count += 1
#                     curr_launch_node, i = retrieval_node, i + len(best_valid_drone_nodes) + 1
#                 else:
#                     seg = RouteSegment(curr_launch_node, truck_route[i])
#                     seg.truck_nodes = []
#                     self.fleet_segments[truck_idx].append(seg)
#                     if truck_route[i].get('road_damaged', 0) == 0 and truck_route[i]['id'] != self.depot['id']:
#                         self.served_count += 1
#                     curr_launch_node, i = truck_route[i], i + 1
                    
#             if curr_launch_node['id'] != self.depot['id']:
#                 self.fleet_segments[truck_idx].append(RouteSegment(curr_launch_node, self.depot))

#     def calculate_route_real_cost(self, segments):
#         travel_cost = 0.0
#         delay_penalty = 0.0
#         curr_time = 0.0
#         for seg in segments:
#             curr_time += seg.duration
#             travel_cost += (seg.dur_ts * TRUCK_COST) + (seg.dur_dt * DRONE_COST)
            
#             sub_time_t = curr_time - seg.duration
#             curr_pos_t = seg.start_node
#             for tn in seg.truck_nodes:
#                 sub_time_t += calc_manhattan_dist(curr_pos_t, tn) / TRUCK_SPEED
#                 deadline = tn.get('deadline', MAX_WORKING_HOURS)
#                 if sub_time_t > deadline: delay_penalty += (sub_time_t - deadline) * DELAY_ALPHA
#                 curr_pos_t = tn
                
#             sub_time_d = curr_time - seg.duration
#             curr_pos_d = seg.start_node
#             for dn in seg.drone_nodes:
#                 sub_time_d += calc_euclidean_dist(curr_pos_d, dn) / DRONE_SPEED
#                 deadline = dn.get('deadline', MAX_WORKING_HOURS)
#                 if sub_time_d > deadline: delay_penalty += (sub_time_d - deadline) * DELAY_ALPHA
#                 curr_pos_d = dn
                
#             end_deadline = seg.end_node.get('deadline', MAX_WORKING_HOURS)
#             if curr_time > end_deadline:
#                 delay_penalty += (curr_time - end_deadline) * DELAY_ALPHA
                
#             if curr_time > MAX_WORKING_HOURS:
#                 delay_penalty += (curr_time - MAX_WORKING_HOURS) * DELAY_ALPHA 
                
#         return travel_cost, delay_penalty

#     def try_insert_truck(self, seg, new_node, insert_idx, current_total_load):
#         # 1. 检查路损
#         if new_node.get('road_damaged', 0) == 1: return False
#         # 2. 🌟 检查卡车全局载重是否超标
#         if current_total_load + new_node.get('demand', 0) > ENV_CFG.get('WT_max', 1000.0): return False
        
#         seg.truck_nodes.insert(insert_idx, new_node)
#         seg._update_stats()
#         # 3. 检查无人机是否会没电
#         if seg.duration > DRONE_MAX_ENDURANCE and seg.drone_nodes: return False
#         return True

#     def try_insert_drone(self, seg, new_node, insert_idx, current_total_load):
#         # 1. 检查无人机单次起飞载重
#         if seg.drone_load + new_node.get('demand', 0) > DRONE_MAX_CAPACITY: return False
#         # 2. 🌟 检查卡车全局载重是否超标 (因为无人机的货也是卡车带的)
#         if current_total_load + new_node.get('demand', 0) > ENV_CFG.get('WT_max', 1000.0): return False
        
#         seg.drone_nodes.insert(insert_idx, new_node)
#         seg._update_stats()
#         # 3. 检查无人机是否会没电
#         if seg.duration > DRONE_MAX_ENDURANCE: return False
#         return True

#     def online_dynamic_insertion(self):
#         accepted, rejected = 0, 0
#         for dyn_node in self.pending_customers:
#             best_delta_cost = float('inf')
#             best_truck_idx, best_seg_idx, best_action = -1, -1, None
            
#             for t_idx, segments in enumerate(self.fleet_segments):
#                 base_travel, base_delay = self.calculate_route_real_cost(segments)
#                 base_cost = base_travel + base_delay
                
#                 # 🌟 计算这辆卡车目前已经装了多少货
#                 current_total_load = 0.0
#                 for s in segments:
#                     current_total_load += sum(n.get('demand', 0) for n in s.truck_nodes)
#                     current_total_load += sum(n.get('demand', 0) for n in s.drone_nodes)
                
#                 for s_idx, seg in enumerate(segments):
#                     # 尝试放入卡车
#                     for i in range(len(seg.truck_nodes) + 1):
#                         test_segments = copy.deepcopy(segments)
#                         # 传入 current_total_load
#                         if self.try_insert_truck(test_segments[s_idx], dyn_node, i, current_total_load):
#                             new_t, new_d = self.calculate_route_real_cost(test_segments)
#                             new_cost = new_t + new_d
#                             if new_cost - base_cost < best_delta_cost:
#                                 best_delta_cost = new_cost - base_cost
#                                 best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'truck', 'idx':i}
                                
#                     # 尝试放入无人机
#                     for i in range(len(seg.drone_nodes) + 1):
#                         test_segments = copy.deepcopy(segments)
#                         # 传入 current_total_load
#                         if self.try_insert_drone(test_segments[s_idx], dyn_node, i, current_total_load):
#                             new_t, new_d = self.calculate_route_real_cost(test_segments)
#                             new_cost = new_t + new_d
#                             if new_cost - base_cost < best_delta_cost:
#                                 best_delta_cost = new_cost - base_cost
#                                 best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'drone', 'idx':i}

#             # 确认接单逻辑 (修复调用)
#             if best_truck_idx != -1 and best_delta_cost < (UNIT_REWARD_P + UNSERVED_PENALTY):
#                 test_segments = copy.deepcopy(self.fleet_segments[best_truck_idx])
                
#                 # 再次计算获胜卡车的当前总载重，供最终执行插入使用
#                 winner_total_load = 0.0
#                 for s in self.fleet_segments[best_truck_idx]:
#                     winner_total_load += sum(n.get('demand', 0) for n in s.truck_nodes)
#                     winner_total_load += sum(n.get('demand', 0) for n in s.drone_nodes)

#                 if best_action['type'] == 'truck':
#                     self.try_insert_truck(test_segments[best_seg_idx], dyn_node, best_action['idx'], winner_total_load)
#                 else:
#                     self.try_insert_drone(test_segments[best_seg_idx], dyn_node, best_action['idx'], winner_total_load)
                
#                 self.fleet_segments[best_truck_idx] = test_segments
#                 self.served_count += 1
#                 accepted += 1
#             else:
#                 rejected += 1

#     def calculate_final_profit(self):
#         travel_cost = 0.0
#         delay_penalty = 0.0
#         makespan = 0.0
        
#         for segments in self.fleet_segments:
#             t_cost, d_pen = self.calculate_route_real_cost(segments)
#             travel_cost += t_cost
#             delay_penalty += d_pen
#             makespan = max(makespan, sum(seg.duration for seg in segments))
            
#         service_reward_sum = self.served_count * UNIT_REWARD_P
#         total_customers = len(self.nodes) - 1 
#         unserved_count = total_customers - self.served_count
#         unserved_penalty_sum = unserved_count * UNSERVED_PENALTY
        
#         real_reward = service_reward_sum - travel_cost - delay_penalty - unserved_penalty_sum
        
#         return {
#             'served_count': self.served_count,
#             'unserved_count': unserved_count,
#             'service_reward_sum': service_reward_sum,
#             'travel_cost': travel_cost,
#             'delay_penalty': delay_penalty,
#             'unserved_penalty': unserved_penalty_sum,
#             'real_reward': real_reward,
#             'time_used': makespan
#         }

#     def print_detailed_schedule(self):
#         pass # 静默模式可忽略

#     def solve(self):
#         all_truck_routes = self.solve_pdptw_with_ortools(self.static_truck_customers)
#         self.reassignment_procedure(all_truck_routes)
#         self.online_dynamic_insertion()
#         stats = self.calculate_final_profit()
#         return stats

# if __name__ == "__main__":
#     import argparse
#     import os
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, default="dynamic_instance.json")
#     args = parser.parse_args()
#     if os.path.exists(args.dataset):
#         GuDynamicSolver(args.dataset).solve()

# import json
# import math
# import copy
# from ortools.constraint_solver import routing_enums_pb2
# from ortools.constraint_solver import pywrapcp

# # ==========================================
# # 动态加载 Config 参数与惩罚机制
# # ==========================================
# from trucks_and_drones.config import cfg

# ENV_CFG = cfg['environment']
# NODE_CFG = cfg['node']

# NUM_TRUCKS = ENV_CFG.get('num_trucks', 5)                
# NUM_DRONES = ENV_CFG.get('num_drones', 5)                
# TRUCK_SPEED = ENV_CFG.get('truck_speed', 0.5)            
# DRONE_SPEED = ENV_CFG.get('drone_speed', 1.0)            
# MAX_WORKING_HOURS = ENV_CFG.get('horizon', 300.0)        
# DRONE_MAX_ENDURANCE = ENV_CFG.get('max_charge', 50.0)    
# DRONE_MAX_CAPACITY = ENV_CFG.get('WD_max', 80.0)         
# TRUCK_COST = ENV_CFG.get('ct_cost', 0.083)               
# DRONE_COST = ENV_CFG.get('cd_cost', 0.021)               

# # 奖励与惩罚参数对齐 RL 环境
# UNIT_REWARD_P = NODE_CFG['customer'].get('beta', 30.0)         
# UNSERVED_PENALTY = NODE_CFG['customer'].get('beta', 30.0)      
# DELAY_ALPHA = NODE_CFG['customer'].get('alpha', 0.5)           

# # 用于 OR-Tools 的资金放大倍数 (将 0.083 这种小数化为整数精度)
# COST_SCALE = 1000  

# # ==========================================
# # 🌟 核心修复 1：严格对齐 RL 的距离测算口径
# # ==========================================
# def calc_manhattan_dist(n1, n2):
#     """卡车走城市路网：曼哈顿距离"""
#     return abs(n1['x'] - n2['x']) + abs(n1['y'] - n2['y'])

# def calc_euclidean_dist(n1, n2):
#     """无人机飞直线：欧式距离"""
#     return math.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])

# class RouteSegment:
#     def __init__(self, start_node, end_node):
#         self.start_node = start_node
#         self.end_node = end_node
#         self.truck_nodes = []  
#         self.drone_nodes = []  
#         self._update_stats()

#     def _update_stats(self):
#         # 卡车采用曼哈顿距离计算
#         t_dist = 0.0
#         curr = self.start_node
#         for n in self.truck_nodes:
#             t_dist += calc_manhattan_dist(curr, n)
#             curr = n
#         t_dist += calc_manhattan_dist(curr, self.end_node)
#         self.dur_ts = t_dist / TRUCK_SPEED

#         # 无人机采用欧式距离计算
#         d_dist = 0.0
#         if not self.drone_nodes:
#             self.dur_dt = 0.0
#         else:
#             curr = self.start_node
#             for n in self.drone_nodes:
#                 d_dist += calc_euclidean_dist(curr, n)
#                 curr = n
#             d_dist += calc_euclidean_dist(curr, self.end_node)
#             self.dur_dt = d_dist / DRONE_SPEED
            
#         self.duration = max(self.dur_ts, self.dur_dt)
#         self.cost_ts = self.dur_ts * TRUCK_COST
#         self.cost_dt = self.dur_dt * DRONE_COST
        
#         self.drone_energy = self.duration if self.drone_nodes else 0.0
#         self.drone_load = sum(n.get('demand', 1) for n in self.drone_nodes)


# class GuDynamicSolver:
#     def __init__(self, dataset_path):
#         with open(dataset_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         self.nodes = {n['id']: n for n in data['nodes']}
#         self.depot = self.nodes[0]
        
#         self.static_truck_customers = [
#             n for n in data['nodes'] 
#             if not n['is_dynamic'] and n['id'] != 0 and n.get('road_damaged', 0) == 0
#         ]
        
#         self.pending_customers = [
#             n for n in data['nodes'] 
#             if (n['is_dynamic'] or n.get('road_damaged', 0) == 1) and n['id'] != 0
#         ]
#         self.pending_customers.sort(key=lambda x: x['release_time'])
        
#         self.fleet_segments = [ [] for _ in range(NUM_TRUCKS) ]
#         self.served_count = 0

#     def solve_pdptw_with_ortools(self, nodes_to_visit):
#         routing_nodes = [self.depot] + nodes_to_visit
#         num_nodes = len(routing_nodes)
        
#         manager = pywrapcp.RoutingIndexManager(num_nodes, NUM_TRUCKS, 0)
#         routing = pywrapcp.RoutingModel(manager)

#         def cost_callback(from_index, to_index):
#             if from_index == to_index: return 0
#             from_node = manager.IndexToNode(from_index)
#             to_node = manager.IndexToNode(to_index)
#             # OR-Tools 底层规划卡车路线，必须用曼哈顿距离
#             travel_time = calc_manhattan_dist(routing_nodes[from_node], routing_nodes[to_node]) / TRUCK_SPEED
#             return int(travel_time * TRUCK_COST * COST_SCALE)

#         cost_callback_index = routing.RegisterTransitCallback(cost_callback)
#         routing.SetArcCostEvaluatorOfAllVehicles(cost_callback_index)

#         # ==========================================
#         # 🌟 修复引擎 2：纯粹的【物理时间回调】
#         # ==========================================
#         def time_callback(from_index, to_index):
#             if from_index == to_index: return 0
#             from_node = manager.IndexToNode(from_index)
#             to_node = manager.IndexToNode(to_index)
#             travel_time = calc_dist(routing_nodes[from_node], routing_nodes[to_node]) / TRUCK_SPEED
#             return int(travel_time * 100) # 时间放大 100 倍以保留精度

#         time_callback_index = routing.RegisterTransitCallback(time_callback)
#         routing.AddDimension(
#             time_callback_index,
#             30000, int(MAX_WORKING_HOURS * 100), False, 'Time')
#         time_dimension = routing.GetDimensionOrDie('Time')
        
#         for i, n in enumerate(routing_nodes):
#             if i == 0: continue
#             index = manager.NodeToIndex(i)
#             start = int(n.get('release_time', 0) * 100)
#             deadline = int(n.get('deadline', MAX_WORKING_HOURS) * 100)
#             max_time = int(MAX_WORKING_HOURS * 100)
            
#             time_dimension.CumulVar(index).SetRange(start, max_time)
            
#             # 🌟 核心对齐：换算真实的延误罚金
#             # 1分钟 = 100刻度. 1分钟延误真实代价 = 0.5 * 1000 = 500.
#             # 所以 1个时间刻度的罚金应该是 = 500 / 100 = 5
#             penalty_rate = int((DELAY_ALPHA * COST_SCALE) / 100)
#             time_dimension.SetCumulVarSoftUpperBound(index, deadline, penalty_rate)

#         # 🌟 真实拒单代价：30 收益 + 30 惩罚 = 60，折算放大 = 60,000
#         real_drop_penalty = int((UNIT_REWARD_P + UNSERVED_PENALTY) * COST_SCALE)
#         for node in range(1, num_nodes):
#             routing.AddDisjunction([manager.NodeToIndex(node)], real_drop_penalty)

#         search_parameters = pywrapcp.DefaultRoutingSearchParameters()
#         search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
#         search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
#         search_parameters.time_limit.seconds = 8

#         solution = routing.SolveWithParameters(search_parameters)
#         if not solution: raise ValueError("OR-Tools 无解！")

#         all_routes = []
#         for vehicle_id in range(NUM_TRUCKS):
#             index = routing.Start(vehicle_id)
#             route_sequence = []
#             while not routing.IsEnd(index):
#                 route_sequence.append(routing_nodes[manager.IndexToNode(index)])
#                 index = solution.Value(routing.NextVar(index))
#             route_sequence.append(routing_nodes[manager.IndexToNode(index)]) 
#             if len(route_sequence) > 2: all_routes.append(route_sequence)
#         return all_routes

#     def reassignment_procedure(self, all_truck_routes):
#         self.fleet_segments = [ [] for _ in range(NUM_TRUCKS) ]
        
#         for truck_idx, truck_route in enumerate(all_truck_routes):
#             curr_launch_node = truck_route[0]
#             i = 1
#             while i < len(truck_route) - 1:
#                 candidate_drone_nodes, retrieval_node, best_valid_drone_nodes = [], None, []
#                 for j in range(i, len(truck_route) - 1):
#                     test_node = truck_route[j]
#                     candidate_drone_nodes.append(test_node)
#                     test_retrieval = truck_route[j + 1]
                    
#                     if test_retrieval.get('road_damaged', 0) == 1:
#                         break
                        
#                     temp_seg = RouteSegment(curr_launch_node, test_retrieval)
#                     temp_seg.drone_nodes = list(candidate_drone_nodes)
#                     temp_seg.truck_nodes = [] 
#                     temp_seg._update_stats()
                    
#                     if temp_seg.drone_load <= DRONE_MAX_CAPACITY and temp_seg.drone_energy <= DRONE_MAX_ENDURANCE:
#                         retrieval_node = test_retrieval
#                         best_valid_drone_nodes = list(candidate_drone_nodes)
#                     else: break 
                        
#                 if retrieval_node:
#                     seg = RouteSegment(curr_launch_node, retrieval_node)
#                     seg.drone_nodes = best_valid_drone_nodes
#                     self.fleet_segments[truck_idx].append(seg)
#                     self.served_count += len(best_valid_drone_nodes)
#                     if retrieval_node['id'] != self.depot['id']: self.served_count += 1
#                     curr_launch_node, i = retrieval_node, i + len(best_valid_drone_nodes) + 1
#                 else:
#                     seg = RouteSegment(curr_launch_node, truck_route[i])
#                     seg.truck_nodes = []
#                     self.fleet_segments[truck_idx].append(seg)
#                     if truck_route[i].get('road_damaged', 0) == 0 and truck_route[i]['id'] != self.depot['id']:
#                         self.served_count += 1
#                     curr_launch_node, i = truck_route[i], i + 1
                    
#             if curr_launch_node['id'] != self.depot['id']:
#                 self.fleet_segments[truck_idx].append(RouteSegment(curr_launch_node, self.depot))

#     def calculate_route_real_cost(self, segments):
#         travel_cost = 0.0
#         delay_penalty = 0.0
#         curr_time = 0.0
#         for seg in segments:
#             curr_time += seg.duration
#             travel_cost += (seg.dur_ts * TRUCK_COST) + (seg.dur_dt * DRONE_COST)
            
#             sub_time_t = curr_time - seg.duration
#             curr_pos_t = seg.start_node
#             for tn in seg.truck_nodes:
#                 sub_time_t += calc_dist(curr_pos_t, tn) / TRUCK_SPEED
#                 if sub_time_t > tn['deadline']: delay_penalty += (sub_time_t - tn['deadline']) * DELAY_ALPHA
#                 curr_pos_t = tn
                
#             sub_time_d = curr_time - seg.duration
#             curr_pos_d = seg.start_node
#             for dn in seg.drone_nodes:
#                 sub_time_d += calc_dist(curr_pos_d, dn) / DRONE_SPEED
#                 if sub_time_d > dn['deadline']: delay_penalty += (sub_time_d - dn['deadline']) * DELAY_ALPHA
#                 curr_pos_d = dn
                
#             if curr_time > seg.end_node['deadline']:
#                 delay_penalty += (curr_time - seg.end_node['deadline']) * DELAY_ALPHA
                
#             if curr_time > MAX_WORKING_HOURS:
#                 delay_penalty += (curr_time - MAX_WORKING_HOURS) * DELAY_ALPHA 
                
#         return travel_cost + delay_penalty

#     def try_insert_truck(self, seg, new_node, insert_idx):
#         if new_node.get('road_damaged', 0) == 1: return False
#         seg.truck_nodes.insert(insert_idx, new_node)
#         seg._update_stats()
#         if seg.duration > DRONE_MAX_ENDURANCE and seg.drone_nodes: return False
#         return True

#     def try_insert_drone(self, seg, new_node, insert_idx):
#         if seg.drone_load + new_node.get('demand', 1) > DRONE_MAX_CAPACITY: return False
#         seg.drone_nodes.insert(insert_idx, new_node)
#         seg._update_stats()
#         if seg.duration > DRONE_MAX_ENDURANCE: return False
#         return True

#     def online_dynamic_insertion(self):
#         accepted, rejected = 0, 0
        
#         for dyn_node in self.pending_customers:
#             best_delta_cost = float('inf')
#             best_truck_idx, best_seg_idx, best_action = -1, -1, None
            
#             for t_idx, segments in enumerate(self.fleet_segments):
#                 base_cost = self.calculate_route_real_cost(segments)
                
#                 for s_idx, seg in enumerate(segments):
#                     # 尝试放入卡车
#                     for i in range(len(seg.truck_nodes) + 1):
#                         test_segments = copy.deepcopy(segments)
#                         if self.try_insert_truck(test_segments[s_idx], dyn_node, i):
#                             new_cost = self.calculate_route_real_cost(test_segments)
#                             if new_cost - base_cost < best_delta_cost:
#                                 best_delta_cost = new_cost - base_cost
#                                 best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'truck', 'idx':i}
                                
#                     # 尝试放入无人机
#                     for i in range(len(seg.drone_nodes) + 1):
#                         test_segments = copy.deepcopy(segments)
#                         if self.try_insert_drone(test_segments[s_idx], dyn_node, i):
#                             new_cost = self.calculate_route_real_cost(test_segments)
#                             if new_cost - base_cost < best_delta_cost:
#                                 best_delta_cost = new_cost - base_cost
#                                 best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'drone', 'idx':i}

#             # 🌟 RL 同步：只要新增代价(耗油+各类延误等) 小于接单获取的回报(收益+免于弃单罚款)
#             # 这就绝对是一笔划算的买卖，哪怕他迟到一点！
#             if best_truck_idx != -1 and best_delta_cost < (UNIT_REWARD_P + UNSERVED_PENALTY):
#                 test_segments = copy.deepcopy(self.fleet_segments[best_truck_idx])
#                 if best_action['type'] == 'truck':
#                     self.try_insert_truck(test_segments[best_seg_idx], dyn_node, best_action['idx'])
#                 else:
#                     self.try_insert_drone(test_segments[best_seg_idx], dyn_node, best_action['idx'])
#                 self.fleet_segments[best_truck_idx] = test_segments
#                 self.served_count += 1
#                 accepted += 1
#             else:
#                 rejected += 1

#     def print_detailed_schedule(self):
#         """
#         打印详细的时刻表，用于核对到达时间与 deadline
#         """
#         print("\n" + "="*60)
#         print("🕒 详细时刻表与时间窗核对 (Debug Log)")
#         print("="*60)
        
#         for t_idx, segments in enumerate(self.fleet_segments):
#             if not segments:
#                 continue
#             print(f"\n🚛 卡车编队 {t_idx} 路线详情:")
#             curr_time = 0.0
            
#             for s_idx, seg in enumerate(segments):
#                 start_time = curr_time
#                 curr_time += seg.duration  # 片段结束时间
                
#                 # 打印卡车中间节点
#                 sub_time_t = start_time
#                 curr_pos_t = seg.start_node
#                 for tn in seg.truck_nodes:
#                     sub_time_t += calc_dist(curr_pos_t, tn) / TRUCK_SPEED
#                     delay = max(0, sub_time_t - tn['deadline'])
#                     status = "✅ 提前到达" if delay == 0 else f"❌ 延误 {delay:.2f}分"
#                     print(f"  ┣━ [卡车送货] 客户 {tn['id']:2d} | 抵达: {sub_time_t:6.2f} | 截止: {tn['deadline']:6.2f} | {status}")
#                     curr_pos_t = tn
                    
#                 # 打印无人机中间节点
#                 sub_time_d = start_time
#                 curr_pos_d = seg.start_node
#                 for dn in seg.drone_nodes:
#                     sub_time_d += calc_dist(curr_pos_d, dn) / DRONE_SPEED
#                     delay = max(0, sub_time_d - dn['deadline'])
#                     status = "✅ 提前到达" if delay == 0 else f"❌ 延误 {delay:.2f}分"
#                     print(f"  ┣━ [无人机送] 客户 {dn['id']:2d} | 抵达: {sub_time_d:6.2f} | 截止: {dn['deadline']:6.2f} | {status}")
#                     curr_pos_d = dn
                    
#                 # 打印片段汇合/终点
#                 if seg.end_node['id'] != self.depot['id']:
#                     delay = max(0, curr_time - seg.end_node['deadline'])
#                     status = "✅ 提前到达" if delay == 0 else f"❌ 延误 {delay:.2f}分"
#                     print(f"  ┗━ [汇合/服务] 客户 {seg.end_node['id']:2d} | 抵达: {curr_time:6.2f} | 截止: {seg.end_node['deadline']:6.2f} | {status}")
#                 else:
#                     print(f"  ┗━ [返回仓库] Depot 0 | 抵达: {curr_time:6.2f} | 截止: {MAX_WORKING_HOURS:6.2f}")

#     def calculate_final_profit(self):
#         total_truck_time = sum(sum(seg.dur_ts for seg in segments) for segments in self.fleet_segments)
#         total_drone_time = sum(sum(seg.dur_dt for seg in segments) for segments in self.fleet_segments)
        
#         total_cost_and_delay = sum(self.calculate_route_real_cost(segments) for segments in self.fleet_segments)
#         travel_cost = (total_truck_time * TRUCK_COST) + (total_drone_time * DRONE_COST)
#         delay_penalty = total_cost_and_delay - travel_cost
        
#         makespan = max((sum(seg.duration for seg in segments) for segments in self.fleet_segments), default=0)
        
#         service_reward_sum = self.served_count * UNIT_REWARD_P
#         total_customers = len(self.nodes) - 1 
#         unserved_count = total_customers - self.served_count
#         unserved_penalty_sum = unserved_count * UNSERVED_PENALTY
        
#         real_reward = service_reward_sum - travel_cost - delay_penalty - unserved_penalty_sum
        
#         return {
#             'served_count': self.served_count,
#             'unserved_count': unserved_count,
#             'service_reward_sum': service_reward_sum,
#             'travel_cost': travel_cost,
#             'delay_penalty': delay_penalty,
#             'unserved_penalty': unserved_penalty_sum,
#             'real_reward': real_reward,
#             'time_used': makespan
#         }

#     def solve(self):
#         all_truck_routes = self.solve_pdptw_with_ortools(self.static_truck_customers)
#         self.reassignment_procedure(all_truck_routes)
#         self.online_dynamic_insertion()
        
#         # 🌟 调用打印时刻表
#         self.print_detailed_schedule()
        
#         stats = self.calculate_final_profit()
        
#         print(f"\n======== 最终 Benchmark 评估结果 ========")
#         print(f"部署车队: {NUM_TRUCKS} 辆卡车, {NUM_DRONES} 架无人机")
#         print(f"总客户数: {len(self.nodes)-1} | 已服务: {stats['served_count']} | 未服务: {stats['unserved_count']}")
#         print(f"----------------------------------------")
#         print(f"💰 服务总收益 (+): {stats['service_reward_sum']:.2f}")
#         print(f"⛽ 行驶总成本 (-): {stats['travel_cost']:.2f}")
#         print(f"⏱️ 延迟总惩罚 (-): {stats['delay_penalty']:.2f}")
#         print(f"❌ 未服务惩罚 (-): {stats['unserved_penalty']:.2f}")
#         print(f"========================================")
#         print(f"🏆 最终净利润 (Real Reward): {stats['real_reward']:.2f}")
        
#         return stats

# if __name__ == "__main__":
#     import argparse
#     import os
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, default="dynamic_instance.json")
#     args = parser.parse_args()

#     if not os.path.exists(args.dataset):
#         print(f"❌ 运行失败: 找不到数据集文件 '{args.dataset}'")
#     else:
#         GuDynamicSolver(args.dataset).solve()