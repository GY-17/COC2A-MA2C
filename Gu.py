import json
import math
import copy
import random # 新增：用于离线采样
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
    return abs(n1['x'] - n2['x']) + abs(n1['y'] - n2['y'])

def calc_euclidean_dist(n1, n2):
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

    def _generate_dummy_opcrs(self, num_nodes):
        """生成符合历史分布的虚拟动态订单，绝不使用真实的 pending_customers"""
        dummy_nodes = []
        spatial_pool = self.static_truck_customers if self.static_truck_customers else [self.depot]
        
        for i in range(num_nodes):
            ref_node = random.choice(spatial_pool)
            release_t = random.uniform(0, ENV_CFG.get('horizon', 300.0) * 0.5)
            
            # 🌟 修复：加入物理距离时间，防止 OR-Tools 报无解
            dist_to_depot = calc_manhattan_dist(ref_node, self.depot)
            travel_time = dist_to_depot / TRUCK_SPEED
            earliest_arrival = release_t + travel_time
            deadline = min(MAX_WORKING_HOURS, earliest_arrival + random.uniform(10, 50))
            
            dummy_node = {
                'id': -1000 - i,  # 负数 ID
                'x': ref_node['x'] + random.uniform(-1, 1), 
                'y': ref_node['y'] + random.uniform(-1, 1),
                'demand': random.uniform(5, 15), 
                'release_time': release_t,
                'deadline': deadline,  # 使用修复后的宽容时间窗
                'road_damaged': 0, 
                'is_dynamic': True
            }
            dummy_nodes.append(dummy_node)
        return dummy_nodes

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

    def reassignment_procedure(self, all_truck_routes, sampled_ids=None):
        """🌟 修改点1: 增加 sampled_ids 参数，强制遵守论文中采样节点不可作为起降点的约束"""
        if sampled_ids is None:
            sampled_ids = set()
            
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
                        
                    # 🌟 论文 4.2.1 约束：采样的节点不能作为起降点
                    if test_retrieval['id'] in sampled_ids:
                        continue 
                        
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
                    curr_launch_node, i = retrieval_node, i + len(best_valid_drone_nodes) + 1
                else:
                    seg = RouteSegment(curr_launch_node, truck_route[i])
                    seg.truck_nodes = []
                    self.fleet_segments[truck_idx].append(seg)
                    curr_launch_node, i = truck_route[i], i + 1
                    
            if curr_launch_node['id'] != self.depot['id']:
                self.fleet_segments[truck_idx].append(RouteSegment(curr_launch_node, self.depot))
                
        self._update_served_count()

    def _update_served_count(self):
        """独立统计实际服务的节点数量，避免计算被抽样的虚假节点"""
        served = set()
        for segments in self.fleet_segments:
            for seg in segments:
                # 🌟 修复：只统计真实节点 (id > 0)，排除负数 ID 的虚拟节点
                if seg.start_node['id'] > 0: served.add(seg.start_node['id'])
                if seg.end_node['id'] > 0: served.add(seg.end_node['id'])
                for n in seg.truck_nodes + seg.drone_nodes:
                    if n['id'] > 0: served.add(n['id'])
        self.served_count = len(served)

    # =========================================================================
    # 🌟 新增区：实现论文 4.2 节的 Offline Enhanced Construction Algorithm (OECA)
    # =========================================================================

    def generate_candidate_route(self, sampled_opcrs):
        """构造候选路线：规划 -> 彻底剔除采样节点 -> 重分配留出弹性"""
        sampled_ids = {n['id'] for n in sampled_opcrs}
        nodes_to_visit = self.static_truck_customers + sampled_opcrs
        
        all_truck_routes = self.solve_pdptw_with_ortools(nodes_to_visit)
        
        # 🌟 核心修复：在分配无人机之前，直接从卡车路线中彻底剔除这些假节点！
        cleaned_truck_routes = []
        for route in all_truck_routes:
            clean_route = [n for n in route if n['id'] not in sampled_ids]
            if len(clean_route) >= 2: # 至少保留了 depot 起终点
                cleaned_truck_routes.append(clean_route)
        
        # 暂时覆盖全局 segments 以供提取
        original_segments = copy.deepcopy(self.fleet_segments)
        self.reassignment_procedure(cleaned_truck_routes, set()) # 既然已剔除，无需传 sampled_ids
        candidate_segments = copy.deepcopy(self.fleet_segments)
        self.fleet_segments = original_segments
                
        return candidate_segments

    def offline_enhanced_construction(self, num_candidates=3, num_simulations=3):
        """执行离线模拟增强，挑选最抗波动的初始路线"""
        candidates = []
        
        # 🌟 修复泄漏点：根据宏观概率计算预期的动态节点数量，彻底消除对当前回合真实信息的窥探
        total_customers = len(self.nodes) - 1 # 排除 Depot
        dod = ENV_CFG.get('dynamic_nodes', {}).get('dod', 0.0)
        damage_ratio = ENV_CFG.get('road_damage_ratio', 0.0)
        
        # 数学期望的动态事件数 = 纯动态节点数 + (静态节点数 * 发生损坏的概率)
        expected_dynamic_count = int(total_customers * dod + total_customers * (1 - dod) * damage_ratio)

        # Candidate 0: Baseline (传统做法，不采样)
        try:
            all_truck_routes = self.solve_pdptw_with_ortools(self.static_truck_customers)
            self.reassignment_procedure(all_truck_routes, set())
            candidates.append(copy.deepcopy(self.fleet_segments))
        except ValueError:
            pass
            
        # 生成基于采样的候选路线
        for _ in range(num_candidates - 1):
            # 🌟 修复：使用期望数量进行 40% 采样，取代原来的 int(len(self.pending_customers) * 0.4)
            k = int(expected_dynamic_count * 0.4) 
            sampled_opcrs = self._generate_dummy_opcrs(k) if k > 0 else []
            try:
                cand = self.generate_candidate_route(sampled_opcrs)
                candidates.append(cand)
            except ValueError:
                continue
            
        # 离线模拟挑选最佳候选
        best_candidate = candidates[0] if candidates else self.fleet_segments
        best_avg_profit = -float('inf')
        
        original_pending = copy.deepcopy(self.pending_customers)
        
        for cand in candidates:
            total_sim_profit = 0
            for _ in range(num_simulations):
                # 🌟 修复：使用期望数量进行 80% 模拟，取代原来的 int(len(original_pending) * 0.8)
                k_sim = int(expected_dynamic_count * 0.8)
                sim_pending = self._generate_dummy_opcrs(k_sim) if k_sim > 0 else []
                sim_pending.sort(key=lambda x: x.get('release_time', 0))
                
                self.fleet_segments = copy.deepcopy(cand)
                self.pending_customers = sim_pending
                self._update_served_count()
                
                self.online_dynamic_insertion()
                stats = self.calculate_final_profit()
                total_sim_profit += stats['real_reward']
                
            avg_profit = total_sim_profit / num_simulations
            if avg_profit > best_avg_profit:
                best_avg_profit = avg_profit
                best_candidate = cand
                
        # 恢复真实的在线环境起跑线
        self.fleet_segments = copy.deepcopy(best_candidate)
        self.pending_customers = original_pending
        self._update_served_count()

    # =========================================================================

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

    def validate_route_capacity(self, segments):
        trip_load = 0.0
        WT_max = ENV_CFG.get('WT_max', 1000.0)
        
        for seg in segments:
            if seg.start_node['id'] == 0:
                trip_load = 0.0
            for n in seg.truck_nodes:
                if n['id'] == 0:
                    trip_load = 0.0
                else:
                    trip_load += n.get('demand', 0)
                if trip_load > WT_max: 
                    return False
            for n in seg.drone_nodes:
                if n['id'] != 0:
                    trip_load += n.get('demand', 0)
                if trip_load > WT_max: 
                    return False
            if seg.end_node['id'] == 0:
                trip_load = 0.0
        return True

    def try_insert_truck(self, test_segments, s_idx, new_node, insert_idx):
        seg = test_segments[s_idx]
        if new_node.get('road_damaged', 0) == 1: return False
        
        seg.truck_nodes.insert(insert_idx, new_node)
        seg._update_stats()
        
        if seg.duration > DRONE_MAX_ENDURANCE and seg.drone_nodes: return False
        if not self.validate_route_capacity(test_segments): return False
        return True

    def try_insert_truck_with_depot(self, test_segments, s_idx, new_node, insert_idx):
        seg = test_segments[s_idx]
        if new_node.get('road_damaged', 0) == 1: return False
        
        seg.truck_nodes.insert(insert_idx, self.depot)
        seg.truck_nodes.insert(insert_idx + 1, new_node)
        seg._update_stats()
        
        if seg.duration > DRONE_MAX_ENDURANCE and seg.drone_nodes: return False
        if not self.validate_route_capacity(test_segments): return False
        return True

    def try_insert_drone(self, test_segments, s_idx, new_node, insert_idx):
        seg = test_segments[s_idx]
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
                    for i in range(len(seg.truck_nodes) + 1):
                        test_segments = copy.deepcopy(segments)
                        if self.try_insert_truck(test_segments, s_idx, dyn_node, i):
                            new_t, new_d = self.calculate_route_real_cost(test_segments)
                            new_cost = new_t + new_d
                            if new_cost - base_cost < best_delta_cost:
                                best_delta_cost = new_cost - base_cost
                                best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'truck', 'idx':i, 'with_depot': False}
                        
                        test_segments_depot = copy.deepcopy(segments)
                        if self.try_insert_truck_with_depot(test_segments_depot, s_idx, dyn_node, i):
                            new_t, new_d = self.calculate_route_real_cost(test_segments_depot)
                            new_cost = new_t + new_d
                            if new_cost - base_cost < best_delta_cost:
                                best_delta_cost = new_cost - base_cost
                                best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'truck', 'idx':i, 'with_depot': True}
                                
                    for i in range(len(seg.drone_nodes) + 1):
                        test_segments = copy.deepcopy(segments)
                        if self.try_insert_drone(test_segments, s_idx, dyn_node, i):
                            new_t, new_d = self.calculate_route_real_cost(test_segments)
                            new_cost = new_t + new_d
                            if new_cost - base_cost < best_delta_cost:
                                best_delta_cost = new_cost - base_cost
                                best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'drone', 'idx':i, 'with_depot': False}

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
                self._update_served_count()
                accepted += 1
            else:
                rejected += 1

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

    def solve(self):
        print(f"      🤖 [Gu Solver] 开始离线弹性路线增强 (OECA)...")
        self.offline_enhanced_construction(num_candidates=3, num_simulations=3)
        
        print(f"      🤖 [Gu Solver] 离线路线构造完毕，开始在线动态插入...")
        self.online_dynamic_insertion()
        stats = self.calculate_final_profit()
        
        print(f"      🏆 [Gu Solver] 求解完成！服务真实客户数: {stats['served_count']}, 总利润: {stats['real_reward']:.2f}\n")
        return stats

if __name__ == "__main__":
    import argparse
    import os
    import pprint  # 引入 pprint 让字典输出更美观

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dynamic_instance.json")
    args = parser.parse_args()
    
    if os.path.exists(args.dataset):
        print(f"🚀 正在运行 Gu 的启发式算法，数据集: {args.dataset} ...\n")
        solver = GuDynamicSolver(args.dataset)
        stats = solver.solve()
        
        print("\n" + "="*40)
        print("🏆 最终运行结果统计 (Final Stats):")
        print("="*40)
        pprint.pprint(stats)
    else:
        print(f"❌ 错误: 找不到数据集文件 '{args.dataset}'！")
        print("请检查文件路径，或者通过 --dataset 参数指定正确的文件位置。")

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
#         t_dist = 0.0
#         curr = self.start_node
#         for n in self.truck_nodes:
#             t_dist += calc_manhattan_dist(curr, n)
#             curr = n
#         t_dist += calc_manhattan_dist(curr, self.end_node)
#         self.dur_ts = t_dist / TRUCK_SPEED

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

#         # 增加卡车载重约束维度 (保证初始静态路线合法)
#         def demand_callback(from_index):
#             from_node = manager.IndexToNode(from_index)
#             return int(routing_nodes[from_node].get('demand', 0))

#         demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
#         routing.AddDimensionWithVehicleCapacity(
#             demand_callback_index,
#             0,  
#             [int(ENV_CFG.get('WT_max', 1000.0))] * NUM_TRUCKS, 
#             True, 
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
#         search_parameters.time_limit.seconds = 10

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

#     # =========================================================================
#     # 🌟 核心修改区：加入真实的载重重置逻辑与回仓补给（Multi-Trip）启发式
#     # =========================================================================

#     def validate_route_capacity(self, segments):
#         """
#         按时间顺序遍历整条线路，模拟卡车行驶中的货物消耗。
#         一旦遇到仓库节点(id == 0)，则代表完成补给，当前载重清零。
#         """
#         trip_load = 0.0
#         WT_max = ENV_CFG.get('WT_max', 1000.0)
        
#         for seg in segments:
#             # 起点若是仓库，清空载重
#             if seg.start_node['id'] == 0:
#                 trip_load = 0.0
            
#             # 累加卡车路径节点消耗
#             for n in seg.truck_nodes:
#                 if n['id'] == 0:
#                     trip_load = 0.0
#                 else:
#                     trip_load += n.get('demand', 0)
#                 if trip_load > WT_max: 
#                     return False
            
#             # 累加无人机路径节点消耗（无人机的货是从卡车上带走的）
#             for n in seg.drone_nodes:
#                 if n['id'] != 0:
#                     trip_load += n.get('demand', 0)
#                 if trip_load > WT_max: 
#                     return False
            
#             # 终点若是仓库，清空载重
#             if seg.end_node['id'] == 0:
#                 trip_load = 0.0
                
#         return True

#     def try_insert_truck(self, test_segments, s_idx, new_node, insert_idx):
#         """常规直接插入策略"""
#         seg = test_segments[s_idx]
#         if new_node.get('road_damaged', 0) == 1: return False
        
#         seg.truck_nodes.insert(insert_idx, new_node)
#         seg._update_stats()
        
#         if seg.duration > DRONE_MAX_ENDURANCE and seg.drone_nodes: return False
#         if not self.validate_route_capacity(test_segments): return False
#         return True

#     def try_insert_truck_with_depot(self, test_segments, s_idx, new_node, insert_idx):
#         """
#         🚀 启发式主动回仓补货策略 (Multi-trip Heuristic)：
#         如果没货了，卡车尝试在送货前强制塞入一个 Node 0 行程，进行补血。
#         """
#         seg = test_segments[s_idx]
#         if new_node.get('road_damaged', 0) == 1: return False
        
#         # 强制塞入仓库节点，再塞入新客户节点
#         seg.truck_nodes.insert(insert_idx, self.depot)
#         seg.truck_nodes.insert(insert_idx + 1, new_node)
#         seg._update_stats()
        
#         if seg.duration > DRONE_MAX_ENDURANCE and seg.drone_nodes: return False
#         if not self.validate_route_capacity(test_segments): return False
#         return True

#     def try_insert_drone(self, test_segments, s_idx, new_node, insert_idx):
#         """无人机插入策略"""
#         seg = test_segments[s_idx]
        
#         # 无人机单次起降的独立载重限制
#         if seg.drone_load + new_node.get('demand', 0) > DRONE_MAX_CAPACITY: return False
        
#         seg.drone_nodes.insert(insert_idx, new_node)
#         seg._update_stats()
        
#         if seg.duration > DRONE_MAX_ENDURANCE: return False
#         if not self.validate_route_capacity(test_segments): return False
#         return True

#     def online_dynamic_insertion(self):
#         accepted, rejected = 0, 0
#         for dyn_node in self.pending_customers:
#             best_delta_cost = float('inf')
#             best_truck_idx, best_seg_idx, best_action = -1, -1, None
            
#             for t_idx, segments in enumerate(self.fleet_segments):
#                 base_travel, base_delay = self.calculate_route_real_cost(segments)
#                 base_cost = base_travel + base_delay
                
#                 for s_idx, seg in enumerate(segments):
#                     # =============== 尝试放入卡车 ===============
#                     for i in range(len(seg.truck_nodes) + 1):
#                         # 方案 A：常规插入
#                         test_segments = copy.deepcopy(segments)
#                         if self.try_insert_truck(test_segments, s_idx, dyn_node, i):
#                             new_t, new_d = self.calculate_route_real_cost(test_segments)
#                             new_cost = new_t + new_d
#                             if new_cost - base_cost < best_delta_cost:
#                                 best_delta_cost = new_cost - base_cost
#                                 best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'truck', 'idx':i, 'with_depot': False}
                        
#                         # 方案 B：顺路回仓补货后插入 (突破单次运力限制)
#                         test_segments_depot = copy.deepcopy(segments)
#                         if self.try_insert_truck_with_depot(test_segments_depot, s_idx, dyn_node, i):
#                             new_t, new_d = self.calculate_route_real_cost(test_segments_depot)
#                             new_cost = new_t + new_d
#                             if new_cost - base_cost < best_delta_cost:
#                                 best_delta_cost = new_cost - base_cost
#                                 best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'truck', 'idx':i, 'with_depot': True}
                                
#                     # =============== 尝试放入无人机 ===============
#                     for i in range(len(seg.drone_nodes) + 1):
#                         test_segments = copy.deepcopy(segments)
#                         if self.try_insert_drone(test_segments, s_idx, dyn_node, i):
#                             new_t, new_d = self.calculate_route_real_cost(test_segments)
#                             new_cost = new_t + new_d
#                             if new_cost - base_cost < best_delta_cost:
#                                 best_delta_cost = new_cost - base_cost
#                                 best_truck_idx, best_seg_idx, best_action = t_idx, s_idx, {'type':'drone', 'idx':i, 'with_depot': False}

#             # =============== 确认并执行接单逻辑 ===============
#             if best_truck_idx != -1 and best_delta_cost < (UNIT_REWARD_P + UNSERVED_PENALTY):
#                 test_segments = copy.deepcopy(self.fleet_segments[best_truck_idx])
                
#                 if best_action['type'] == 'truck':
#                     if best_action.get('with_depot'):
#                         self.try_insert_truck_with_depot(test_segments, best_seg_idx, dyn_node, best_action['idx'])
#                     else:
#                         self.try_insert_truck(test_segments, best_seg_idx, dyn_node, best_action['idx'])
#                 else:
#                     self.try_insert_drone(test_segments, best_seg_idx, dyn_node, best_action['idx'])
                
#                 self.fleet_segments[best_truck_idx] = test_segments
#                 self.served_count += 1
#                 accepted += 1
#             else:
#                 rejected += 1

#     # =========================================================================

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
#         pass

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

