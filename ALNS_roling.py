import json
import math
import copy
import random
import numpy as np
from trucks_and_drones.config import cfg

# ================= 配置参数 =================
ENV_CFG = cfg['environment']
NUM_TRUCKS = ENV_CFG.get('num_trucks', 5)                
NUM_DRONES = ENV_CFG.get('num_drones', 5)                
TRUCK_SPEED = ENV_CFG.get('truck_speed', 0.5)            
DRONE_SPEED = ENV_CFG.get('drone_speed', 1.0)            
MAX_WORKING_HOURS = ENV_CFG.get('horizon', 300.0)        
DRONE_MAX_BATTERY = ENV_CFG.get('max_charge', 50.0)    
DRONE_MAX_CAPACITY = ENV_CFG.get('WD_max', 80.0)
TRUCK_MAX_CAPACITY = ENV_CFG.get('WT_max', 1000.0)
TRUCK_COST = ENV_CFG.get('ct_cost', 0.083)               
DRONE_COST = ENV_CFG.get('cd_cost', 0.021)

if 'node' in cfg and 'customer' in cfg['node']:
    UNIT_REWARD_P = cfg['node']['customer'].get('beta', 30.0)         
    UNSERVED_PENALTY = cfg['node']['customer'].get('beta', 30.0)      
    DELAY_ALPHA = cfg['node']['customer'].get('alpha', 0.5)
else:
    UNIT_REWARD_P, UNSERVED_PENALTY, DELAY_ALPHA = 30.0, 30.0, 0.5

class CrossTruckALNS_Rolling:
    def __init__(self, instance_json_path):
        print(f"📂 [初始化] 加载跨车协同数据集: {instance_json_path}")
        with open(instance_json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.nodes = self.data['nodes']
        self.num_nodes = len(self.nodes)
        self.damaged_nodes = set(self.data.get('damaged_nodes', []))
        
        self.dist_T = np.zeros((self.num_nodes, self.num_nodes))
        self.dist_D = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                c1, c2 = np.array([self.nodes[i]['x'], self.nodes[i]['y']]), np.array([self.nodes[j]['x'], self.nodes[j]['y']])
                self.dist_T[i][j] = float(np.sum(np.abs(c1 - c2)))
                self.dist_D[i][j] = float(np.linalg.norm(c1 - c2))

        self.stats = {'total_reward': 0.0, 'travel_cost': 0.0, 'delay_penalty': 0.0, 
                      'truck_distance': 0.0, 'drone_distance': 0.0, 'served_nodes': set()}

    def evaluate_cross_routes(self, truck_routes, drone_sorties, truck_start_times, truck_start_payloads):
        drone_dist = 0.0
        
        # 1. 载重验证
        payload_drops = [[0.0] * len(truck_routes[k]) for k in range(NUM_TRUCKS)]
        for k in range(NUM_TRUCKS):
            for i in range(1, len(truck_routes[k])):
                node = truck_routes[k][i]
                if node != 0: 
                    payload_drops[k][i] += self.nodes[node]['demand']
                    
        # 【修复1】：支持多目标点，严格校验电池与无人机总载重
        for d, t1, L, tgts, t2, R in drone_sorties:
            total_demand = sum(self.nodes[tgt]['demand'] for tgt in tgts)
            if total_demand > DRONE_MAX_CAPACITY: 
                return float('inf'), 0, 0, None, False
            
            flight_dist = 0.0
            curr_pos = truck_routes[t1][L]
            for tgt in tgts:
                flight_dist += self.dist_D[curr_pos][tgt]
                curr_pos = tgt
            flight_dist += self.dist_D[curr_pos][truck_routes[t2][R]]
            
            if flight_dist > DRONE_MAX_BATTERY: 
                return float('inf'), 0, 0, None, False
                
            drone_dist += flight_dist
            payload_drops[t1][L] += total_demand

        # 卡车载重验证
        for k in range(NUM_TRUCKS):
            cap = truck_start_payloads[k]
            for i in range(len(truck_routes[k])):
                if truck_routes[k][i] == 0 and i > 0: cap = TRUCK_MAX_CAPACITY
                cap -= payload_drops[k][i]
                if cap < -1e-5: return float('inf'), 0, 0, None, False

        # 2. 时间同步 
        node_times = [[0.0] * len(truck_routes[k]) for k in range(NUM_TRUCKS)]
        for k in range(NUM_TRUCKS): node_times[k][0] = truck_start_times[k]
        
        changed = True
        iterations = 0
        while changed and iterations < 5:
            changed = False
            iterations += 1
            # 卡车沿路推进
            for k in range(NUM_TRUCKS):
                for i in range(1, len(truck_routes[k])):
                    prev, curr = truck_routes[k][i-1], truck_routes[k][i]
                    arr_t = node_times[k][i-1] + (self.dist_T[prev][curr] / TRUCK_SPEED)
                    if arr_t > node_times[k][i] + 1e-5:
                        node_times[k][i] = arr_t
                        changed = True
            
            # 无人机约束：回收点的时间必须 >= 起飞时间 + 连续飞行时间
            for d, t1, L, tgts, t2, R in drone_sorties:
                launch_t = node_times[t1][L]
                
                flight_dist = 0.0
                curr_pos = truck_routes[t1][L]
                for tgt in tgts:
                    flight_dist += self.dist_D[curr_pos][tgt]
                    curr_pos = tgt
                flight_dist += self.dist_D[curr_pos][truck_routes[t2][R]]
                
                flight_time = flight_dist / DRONE_SPEED
                land_t = launch_t + flight_time
                if land_t > node_times[t2][R] + 1e-5:
                    node_times[t2][R] = land_t 
                    changed = True

        if iterations >= 5: return float('inf'), 0, 0, None, False 

        # 【修复2】：严格无人机资源追踪，防重叠
        drone_intervals = {d: [] for d in range(NUM_DRONES)}
        for d, t1, L, tgts, t2, R in drone_sorties:
            launch_t = node_times[t1][L]
            flight_dist = sum(self.dist_D[prev][curr] for prev, curr in zip([truck_routes[t1][L]] + tgts[:-1], tgts))
            flight_dist += self.dist_D[tgts[-1]][truck_routes[t2][R]]
            land_t = launch_t + (flight_dist / DRONE_SPEED)
            drone_intervals[d].append((launch_t, land_t))
            
        for d, intervals in drone_intervals.items():
            intervals.sort()
            for i in range(1, len(intervals)):
                if intervals[i][0] < intervals[i-1][1] - 1e-5:
                    return float('inf'), 0, 0, None, False # 无人机分身乏术，冲突！
        
        # 3. 成本计算
        cost, truck_dist = drone_dist * DRONE_COST, 0.0
        served_nodes = set()
        
        for k in range(NUM_TRUCKS):
            for i in range(1, len(truck_routes[k])):
                node = truck_routes[k][i]
                if node != 0: served_nodes.add(node)
                d_t = self.dist_T[truck_routes[k][i-1]][node]
                truck_dist += d_t
                cost += d_t * TRUCK_COST
                deadline = self.nodes[node].get('deadline', MAX_WORKING_HOURS)
                if node_times[k][i] > deadline: 
                    cost += (node_times[k][i] - deadline) * DELAY_ALPHA
                    
        for d, t1, L, tgts, t2, R in drone_sorties:
            curr_pos = truck_routes[t1][L]
            curr_t = node_times[t1][L]
            for tgt in tgts:
                served_nodes.add(tgt)
                curr_t += self.dist_D[curr_pos][tgt] / DRONE_SPEED
                curr_pos = tgt
                deadline = self.nodes[tgt].get('deadline', MAX_WORKING_HOURS)
                if curr_t > deadline:
                    cost += (curr_t - deadline) * DELAY_ALPHA
                
        for node in served_nodes:
            cost -= self.nodes[node].get('beta', UNIT_REWARD_P)
                    
        return cost, truck_dist, drone_dist, node_times, True


    def _repair_cross(self, t_routes, sorties, unassigned, t_times, t_payloads):
        """【性能飞跃版】：O(1) 增量估算 + 延迟验证机制"""
        routes = [r.copy() for r in t_routes]
        s_list = [(d, t1, L, tgts.copy(), t2, R) for (d, t1, L, tgts, t2, R) in sorties]
        unassign = list(unassigned)
        
        while unassign:
            best_regret = -float('inf')
            best_insert_move = None
            best_insert_node = None
            
            # 🔥 [核心修改 1]：引入局部候选池采样
            # 不再遍历全部数百个节点，而是随机抽取最多 15 个节点进行尝试
            # 这将 O(N^4) 的复杂度瞬间降维到 O(N^3) 以下
            eval_pool = random.sample(unassign, min(len(unassign), 15))

            for node in eval_pool:
                is_damaged = node in self.damaged_nodes
                node_demand = self.nodes[node]['demand']
                node_reward = self.nodes[node].get('beta', UNIT_REWARD_P)
                
                # 用于存储 O(1) 估算的候选动作: (估算增量成本, 动作元组)
                insert_candidates = [] 
                
                # ==========================================
                # 阶段 1：O(1) 增量成本估算 (不拷贝，不全量评估)
                # ==========================================
                
                # 策略 A：分配给卡车
                if not is_damaged and node_demand <= TRUCK_MAX_CAPACITY:
                    for k in range(NUM_TRUCKS):
                        for i in range(1, len(routes[k]) + 1):
                            prev_node = routes[k][i-1]
                            next_node = routes[k][i] if i < len(routes[k]) else None
                            
                            # A1. 直接插入当前两点之间
                            if next_node is not None:
                                delta_dist = self.dist_T[prev_node][node] + self.dist_T[node][next_node] - self.dist_T[prev_node][next_node]
                            else:
                                delta_dist = self.dist_T[prev_node][node]
                            
                            est_delta_cost = delta_dist * TRUCK_COST - node_reward
                            insert_candidates.append((est_delta_cost, ('T', k, i)))
                            
                            # A2. 中途回仓库补给再服务
                            if prev_node != 0:
                                if next_node is not None:
                                    delta_dist_depot = self.dist_T[prev_node][0] + self.dist_T[0][node] + self.dist_T[node][next_node] - self.dist_T[prev_node][next_node]
                                else:
                                    delta_dist_depot = self.dist_T[prev_node][0] + self.dist_T[0][node]
                                est_delta_cost_depot = delta_dist_depot * TRUCK_COST - node_reward
                                insert_candidates.append((est_delta_cost_depot, ('T_DEPOT', k, i)))
                
                # 策略 B：分配给无人机
                if node_demand <= DRONE_MAX_CAPACITY:
                    # B1. 创立全新的起降任务
                    for t1 in range(NUM_TRUCKS):
                        for L in range(len(routes[t1])):
                            n_launch = routes[t1][L]
                            dist_launch_node = self.dist_D[n_launch][node]
                            if dist_launch_node > DRONE_MAX_BATTERY: continue
                            
                            for t2 in range(NUM_TRUCKS):
                                # 💥 新增剪枝：限制降落点 R 与起飞点 L 的序列差值不超过 5，防止时空错乱
                                window_start = max(0, L - 5) if t1 != t2 else L + 1
                                window_end = min(len(routes[t2]), L + 6)
                                
                                for R in range(window_start, window_end):
                                    n_land = routes[t2][R]
                                    dist_node_land = self.dist_D[node][n_land]
                                    
                                    flight_dist = dist_launch_node + dist_node_land
                                    if flight_dist > DRONE_MAX_BATTERY: continue
                                    
                                    est_delta_cost = flight_dist * DRONE_COST - node_reward
                                    insert_candidates.append((est_delta_cost, ('D_NEW', t1, L, t2, R))) 
                    # B2. 追加到现有的无人机多点任务中
                    for s_idx, (d, t1, L, tgts, t2, R) in enumerate(s_list):
                        current_demand = sum(self.nodes[tgt]['demand'] for tgt in tgts)
                        if current_demand + node_demand > DRONE_MAX_CAPACITY: continue
                            
                        curr_pos = routes[t1][L]
                        current_dist = 0.0
                        for tgt in tgts:
                            current_dist += self.dist_D[curr_pos][tgt]
                            curr_pos = tgt
                        current_dist += self.dist_D[curr_pos][routes[t2][R]]

                        for pos in range(len(tgts) + 1):
                            prev_node = routes[t1][L] if pos == 0 else tgts[pos-1]
                            next_node = routes[t2][R] if pos == len(tgts) else tgts[pos]
                            
                            delta_dist = self.dist_D[prev_node][node] + self.dist_D[node][next_node] - self.dist_D[prev_node][next_node]
                            if current_dist + delta_dist > DRONE_MAX_BATTERY: continue
                            
                            est_delta_cost = delta_dist * DRONE_COST - node_reward
                            insert_candidates.append((est_delta_cost, ('D_APPEND', s_idx, pos)))

                # ==========================================
                # 阶段 2：延迟验证 (Lazy Evaluation)
                # ==========================================
                # 对所有估算成本从小到大排序
                # insert_candidates.sort(key=lambda x: x[0])
                
                # valid_costs = [] # 记录真正合法的插入方案 (绝对成本 n_cost, 动作包)
                
                # # 仅对排名前列的候选进行昂贵的全量评估，拿到前 2 个合法的即可用于计算后悔值
                # for est_cost, move in insert_candidates:
                #     if len(valid_costs) >= 2: 
                #         break 
                        
                #     # 为当前验证构建临时的深拷贝环境
                #     new_r = [r.copy() for r in routes]
                #     new_s = []
                # ==========================================
                # 阶段 2：延迟验证 (Lazy Evaluation) 分类安全版
                # ==========================================
                insert_candidates.sort(key=lambda x: x[0])
                
                valid_costs = [] 
                eval_drones = 0  # 无人机动作计数器
                eval_trucks = 0  # 卡车动作计数器
                
                for est_cost, move in insert_candidates:
                    if len(valid_costs) >= 2: 
                        break 
                        
                    # 💥 核心安全锁：分类限制验证次数
                    is_drone_move = 'D' in move[0]
                    if is_drone_move:
                        eval_drones += 1
                        if eval_drones > 10: continue # 超过10个无人机动作则跳过，寻找卡车动作
                    else:
                        eval_trucks += 1
                        if eval_trucks > 5: continue  # 超过5个卡车动作则跳过
                        
                    # 为当前验证构建临时的深拷贝环境
                    new_r = [r.copy() for r in routes]
                    new_s = []
                    
                    if move[0] == 'T':
                        _, k, i = move
                        new_r[k].insert(i, node)
                        for sd, str1, sL, stgts, str2, sR in s_list:
                            nL = sL + 1 if (str1 == k and sL >= i) else sL
                            nR = sR + 1 if (str2 == k and sR >= i) else sR
                            new_s.append((sd, str1, nL, stgts.copy(), str2, nR))
                            
                    elif move[0] == 'T_DEPOT':
                        _, k, i = move
                        new_r[k].insert(i, 0)
                        new_r[k].insert(i+1, node)
                        for sd, str1, sL, stgts, str2, sR in s_list:
                            nL = sL + 2 if (str1 == k and sL >= i) else sL
                            nR = sR + 2 if (str2 == k and sR >= i) else sR
                            new_s.append((sd, str1, nL, stgts.copy(), str2, nR))
                            
                    elif move[0] == 'D_NEW':
                        _, t1, L, t2, R = move
                        new_s = [(sd, str1, sL, stgts.copy(), str2, sR) for (sd, str1, sL, stgts, str2, sR) in s_list]
                        assigned = False
                        # 找到一架空闲无人机即可
                        for test_d in range(NUM_DRONES):
                            new_s_test = new_s + [(test_d, t1, L, [node], t2, R)]
                            n_cost, _, _, _, val = self.evaluate_cross_routes(new_r, new_s_test, t_times, t_payloads)
                            if val:
                                valid_costs.append((n_cost, ('D_NEW', test_d, None, new_r, new_s_test)))
                                assigned = True
                                break
                        if assigned: continue
                        else: continue # 若没有可用无人机或时间冲突，测试下一个候选
                        
                    elif move[0] == 'D_APPEND':
                        _, s_idx, pos = move
                        new_s = [(sd, str1, sL, stgts.copy(), str2, sR) for (sd, str1, sL, stgts, str2, sR) in s_list]
                        new_s[s_idx][3].insert(pos, node)
                    
                    # 针对非 D_NEW 策略执行全量评估
                    if move[0] != 'D_NEW':
                        n_cost, _, _, _, val = self.evaluate_cross_routes(new_r, new_s, t_times, t_payloads)
                        if val:
                            if move[0] in ['T', 'T_DEPOT']:
                                valid_costs.append((n_cost, (move[0], move[1], move[2], new_r, new_s)))
                            else: # D_APPEND
                                d_id = new_s[s_idx][0]
                                valid_costs.append((n_cost, ('D_APPEND', d_id, None, new_r, new_s)))

                # ==========================================
                # 阶段 3：计算后悔值
                # ==========================================
                if not valid_costs: continue 
                
                best_c = valid_costs[0][0]
                sec_c = valid_costs[1][0] if len(valid_costs) > 1 else best_c + 99999.0
                
                regret = sec_c - best_c
                if regret > best_regret:
                    best_regret = regret
                    best_insert_move = valid_costs[0][1]
                    best_insert_node = node
                                        
            if best_insert_node is not None:
                routes, s_list = best_insert_move[3], best_insert_move[4]
                unassign.remove(best_insert_node)
            else: 
                break
                
        return routes, s_list

    def _destroy_cross(self, routes, sorties, degree=0.3):
        # new_r, new_s = [r.copy() for r in routes], copy.deepcopy(sorties)
        # 修改后 (替换为推导式)
        new_r = [r.copy() for r in routes]
        new_s = [(d, t1, L, tgts.copy(), t2, R) for (d, t1, L, tgts, t2, R) in sorties]
        assigned = []
        for k in range(NUM_TRUCKS): 
            for node in new_r[k][1:]:
                if node != 0: assigned.append(node)
        for s in new_s: 
            assigned.extend(s[3]) # tgts 是个列表了
            
        if not assigned: return new_r, new_s, []
        removed = random.sample(assigned, max(1, int(len(assigned) * degree)))
        
        final_r = [[new_r[k][0]] for k in range(NUM_TRUCKS)]
        final_s = []
        idx_map = [{} for _ in range(NUM_TRUCKS)]
        
        for k in range(NUM_TRUCKS):
            idx_map[k][0] = 0
            new_idx = 1
            for old_idx, node in enumerate(new_r[k][1:], start=1):
                if node not in removed: 
                    final_r[k].append(node)
                    idx_map[k][old_idx] = new_idx
                    new_idx += 1
                else:
                    idx_map[k][old_idx] = -1 
                    
        for d, t1, L, tgts, t2, R in new_s:
            new_tgts = [tgt for tgt in tgts if tgt not in removed]
            if not new_tgts: continue
            
            nL = idx_map[t1].get(L, -1)
            nR = idx_map[t2].get(R, -1)
            
            if nL == -1 or nR == -1:
                removed.extend(new_tgts) 
                continue
                
            final_s.append((d, t1, nL, new_tgts, t2, nR))
            
        return final_r, final_s, list(set(removed))

    def solve(self):
        print("\n🚀 [启动] Cross-Truck 滚动推演主流程...")
        t_locs, t_times = [0] * NUM_TRUCKS, [0.0] * NUM_TRUCKS
        t_payloads = [TRUCK_MAX_CAPACITY] * NUM_TRUCKS 
        current_time = 0.0
        
        while current_time < MAX_WORKING_HOURS:
            known_unserved = [n['id'] for n in self.nodes if n['id'] not in self.stats['served_nodes'] 
                              and n['id'] != 0 and n.get('release_time', 0.0) <= current_time]
            
            if not known_unserved:
                future_times = [n.get('release_time', 0.0) for n in self.nodes if n['id'] not in self.stats['served_nodes'] and n.get('release_time', 0.0) > current_time]
                if not future_times: break 
                current_time = min(future_times)
                continue
                
            idle_trucks = [k for k in range(NUM_TRUCKS) if t_times[k] <= current_time]
            if not idle_trucks:
                current_time = min([t for t in t_times if t > current_time])
                continue

            curr_routes = [[t_locs[k]] for k in range(NUM_TRUCKS)]
            curr_sorties = []
            
            best_routes, best_sorties = self._repair_cross(curr_routes, curr_sorties, known_unserved, t_times, t_payloads)
            best_cost, _, _, best_times, _ = self.evaluate_cross_routes(best_routes, best_sorties, t_times, t_payloads)
            
            if known_unserved:
                print(f"\n      🔥 时间 {current_time:.1f} | 待服务订单: {len(known_unserved)}个 | 重规划开始")
            
            temp = 100.0
            curr_sol_r, curr_sol_s, curr_cost = copy.deepcopy(best_routes), copy.deepcopy(best_sorties), best_cost
            
            # for iter_idx in range(10): 
            # 🔥 [核心修改 2]：动态 ALNS 迭代次数
            # 在 250 节点的初期大幅减少迭代，当节点逐渐被消耗（例如降到 80 以下）时再增加迭代压榨成本
            iter_limit = 5 if len(known_unserved) > 150 else (15 if len(known_unserved) > 80 else 10)
            
            for iter_idx in range(iter_limit): # <- 注意这里从 10 改为了 iter_limit
                dest_r, dest_s, removed = self._destroy_cross(curr_sol_r, curr_sol_s, 0.3)
                new_r, new_s = self._repair_cross(dest_r, dest_s, removed, t_times, t_payloads)
                n_cost, _, _, n_times, val = self.evaluate_cross_routes(new_r, new_s, t_times, t_payloads)
                
                if val:
                    if n_cost < best_cost:
                        best_cost, best_routes, best_sorties, best_times = n_cost, copy.deepcopy(new_r), copy.deepcopy(new_s), copy.deepcopy(n_times)
                        curr_sol_r, curr_sol_s, curr_cost = copy.deepcopy(new_r), copy.deepcopy(new_s), n_cost
                        print(f"      🌟 ALNS迭代 {iter_idx:02d}: 发现更优路线, 成本降至 -> {best_cost:.2f}")
                    elif random.random() < math.exp(-(n_cost - curr_cost) / max(temp, 0.01)):
                        curr_sol_r, curr_sol_s, curr_cost = copy.deepcopy(new_r), copy.deepcopy(new_s), n_cost
                temp *= 0.9

            
            action_taken = False
            for k in idle_trucks:
                rel_sorties = [s for s in best_sorties if (s[1]==k and s[2]==0) or (s[4]==k and s[5]==0)]
                
                # 🌟 新增：跟踪这辆特定卡车是否在本次规划中领到了任务
                truck_acted = False  

                if len(best_routes[k]) > 1:
                    next_node = best_routes[k][1]
                    end_time = best_times[k][1]
                    t_dist = self.dist_T[t_locs[k]][next_node]
                    
                    self.stats['truck_distance'] += t_dist
                    self.stats['travel_cost'] += t_dist * TRUCK_COST
                    
                    if next_node == 0:
                        t_payloads[k] = TRUCK_MAX_CAPACITY
                        print(f"🕒 Time {current_time:.1f}: 🚚 卡车 {k} 回收补给 -> 仓库 0 (预计到 {end_time:.1f})")
                    else:
                        t_payloads[k] -= self.nodes[next_node]['demand']
                        print(f"🕒 Time {current_time:.1f}: 🚚 卡车 {k} 移动 -> 客户 {next_node} [余量:{t_payloads[k]:.1f}kg] (预计到 {end_time:.1f})")
                        if next_node not in self.stats['served_nodes']:
                            self.stats['served_nodes'].add(next_node)
                            self.stats['total_reward'] += UNIT_REWARD_P
                            
                    for d, t1, L, tgts, t2, R in best_sorties:
                        if t1 == k and L == 0:
                            total_dem = sum(self.nodes[tgt]['demand'] for tgt in tgts)
                            t_payloads[k] -= total_dem 
                            print(f"     🚁 释放无人机 -> 客户 {tgts} [扣卡车载重:{total_dem:.1f}kg] (交由卡车 {t2} 回收)")
                            
                            d_dist = 0.0
                            curr_pos = t_locs[k]
                            for tgt in tgts:
                                d_dist += self.dist_D[curr_pos][tgt]
                                curr_pos = tgt
                            d_dist += self.dist_D[curr_pos][best_routes[t2][R]]
                            
                            self.stats['drone_distance'] += d_dist
                            self.stats['travel_cost'] += d_dist * DRONE_COST
                            
                            for tgt in tgts:
                                if tgt not in self.stats['served_nodes']:
                                    self.stats['served_nodes'].add(tgt)
                                    self.stats['total_reward'] += UNIT_REWARD_P
                        if t2 == k and R == 1:
                            print(f"     🛬 回收无人机于 {next_node}")

                    t_locs[k] = next_node
                    t_times[k] = end_time
                    action_taken = True
                    truck_acted = True  # 🌟 标记：此卡车已行动
                    
                else:
                    if rel_sorties:
                        end_time = best_times[k][0]
                        if end_time > t_times[k] + 1e-4:
                            print(f"🕒 Time {current_time:.1f}: 🚚 卡车 {k} 原地等待 -> 调度无人机 (预计到 {end_time:.1f})")
                            for d, t1, L, tgts, t2, R in rel_sorties:
                                if t1 == k and L == 0:
                                    total_dem = sum(self.nodes[tgt]['demand'] for tgt in tgts)
                                    t_payloads[k] -= total_dem
                                    print(f"     🚁 释放无人机 -> 客户 {tgts} [余量:{t_payloads[k]:.1f}kg]")
                                    
                                    d_dist = 0.0
                                    curr_pos = t_locs[k]
                                    for tgt in tgts:
                                        d_dist += self.dist_D[curr_pos][tgt]
                                        curr_pos = tgt
                                    d_dist += self.dist_D[curr_pos][best_routes[t2][R]]
                                    
                                    self.stats['drone_distance'] += d_dist
                                    self.stats['travel_cost'] += d_dist * DRONE_COST
                                    for tgt in tgts:
                                        if tgt not in self.stats['served_nodes']:
                                            self.stats['served_nodes'].add(tgt)
                                            self.stats['total_reward'] += UNIT_REWARD_P
                                if t2 == k and R == 0:
                                    print(f"     🛬 回收无人机于 {t_locs[k]}")
                            t_times[k] = end_time
                            action_taken = True
                            truck_acted = True  # 🌟 标记：此卡车已行动
                    else:
                        if t_locs[k] != 0 and t_payloads[k] < TRUCK_MAX_CAPACITY * 0.5:
                            t_dist = self.dist_T[t_locs[k]][0]
                            t_times[k] += t_dist / TRUCK_SPEED
                            t_locs[k] = 0
                            t_payloads[k] = TRUCK_MAX_CAPACITY
                            self.stats['truck_distance'] += t_dist
                            self.stats['travel_cost'] += t_dist * TRUCK_COST
                            print(f"🕒 Time {current_time:.1f}: 🚚 卡车 {k} 载重告警/空闲 -> 自动返航补给 (预计到 {t_times[k]:.1f})")
                            action_taken = True
                            truck_acted = True  # 🌟 标记：此卡车已行动

                # ==========================================================
                # 🌟 核心修复逻辑：如果该卡车在本次规划中分配到了 0 动作
                # 必须将其时间推延到系统的下一个事件点，避免它卡死在 current_time
                # 造成无限死循环或重复的无意义重规划。
                # ==========================================================
                if not truck_acted:
                    # 必须严格大于当前时间 (加一个极微小的容差 1e-4)，防止同时间点无限循环
                    future_evts = [t for t in t_times if t > current_time + 1e-4]
                    unserved_all = [n.get('release_time', 0.0) for n in self.nodes if n['id'] not in self.stats['served_nodes'] and n['id'] != 0]
                    future_releases = [t for t in unserved_all if t > current_time + 1e-4]
                    future_evts.extend(future_releases)
                    
                    if future_evts:
                        next_evt_time = min(future_evts)
                        # 核心修复：直接对齐到下一个真实事件时间，彻底取消 +0.1
                        t_times[k] = next_evt_time
                    else:
                        t_times[k] = MAX_WORKING_HOURS # 没有未来事件，直接下班
            

    def calculate_final_profit(self):
        all_node_ids = set([n['id'] for n in self.nodes if n['id'] != 0])
        unserved = list(all_node_ids - self.stats['served_nodes'])
        unserved_penalty = len(unserved) * UNSERVED_PENALTY
        self.stats['real_reward'] = self.stats['total_reward'] - self.stats['travel_cost'] - self.stats['delay_penalty'] - unserved_penalty
        
        print(f"\n======== 🏆 Level-3 Cross-Truck ALNS 最终结果 ========")
        print(f"💰 最终净利润: {self.stats['real_reward']:.2f}")
        print(f"🚚 卡车真实行驶: {self.stats['truck_distance']:.1f} km")
        print(f"🚁 无人机真实飞行: {self.stats['drone_distance']:.1f} km")
        print(f"✅ 服务完成率: {len(self.stats['served_nodes'])}/{self.num_nodes-1}")
        return self.stats

if __name__ == "__main__":
    JSON_PATH = "dynamic_instance.json" 
    try:
        solver = CrossTruckALNS_Rolling(JSON_PATH)
        solver.solve()
        solver.calculate_final_profit()
    except Exception as e:
        print(f"❌ 运行报错: {e}")

