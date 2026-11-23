import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
import json
import time

def solve_flexible_vrpd_upper_bound(json_path, config, demands, is_damaged):
    """
    求解【灵活跨车协同】(Flexible Cross-Truck Rendezvous) 的多卡车-多无人机 MIP 模型
    本版本已修改为采用 Peng et al. (2025) 的“灾难性”大 M 法建模方式
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    coords = np.array(data['coords_km'])
    deadlines = data.get('deadlines', [config['horizon']] * len(coords))
    # 🌟 新增：RL 中的 alpha 系数（可手动设为0.5，或传参）
    alpha_penalty = 0.5
    N = len(coords)
    nodes = list(range(N))
    customers = list(range(1, N))
    depot = 0
    
    # 获取参数
    num_trucks = config['num_trucks']
    num_drones = config['num_drones']
    v_t, v_d = config['truck_speed'], config['drone_speed']
    c_t, c_d = config['ct_cost'], config['cd_cost']
    Q_T, Q_D = config['WT_max'], config['WD_max']
    max_charge = config['max_charge']
    horizon = config['horizon']
    reward_val = 30.0 
    
    # 距离与时间矩阵
    time_T = np.zeros((N, N))
    time_D = np.zeros((N, N))
    for i in nodes:
        for j in nodes:
            dist_manhattan = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
            dist_euclidean = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            time_T[i, j] = dist_manhattan / v_t
            time_D[i, j] = dist_euclidean / v_d

    # ==========================
    # Gurobi 模型构建
    # ==========================
    model = gp.Model("Flexible_VRPD_Peng_Style")
    
    # --------------------------
    # 1. 变量生成 (复刻 Peng 论文的无剪枝灾难)
    # --------------------------
    x = model.addVars(nodes, nodes, num_trucks, vtype=GRB.BINARY, name="x")
    
    # 【灾难修改 1】：完全移除 Python 层面的电量过滤，让模型穷举所有节点组合
    y_keys = [(i, j, h, m) for i in nodes for j in customers for h in nodes for m in range(num_drones)
              if i != j and j != h and i != h]
    y = model.addVars(y_keys, vtype=GRB.BINARY, name="y")

    # 逻辑关系变量
    launch_by = model.addVars(nodes, num_trucks, num_drones, vtype=GRB.BINARY, name="launch_by")
    recv_by = model.addVars(nodes, num_trucks, num_drones, vtype=GRB.BINARY, name="recv_by")
    served = model.addVars(customers, vtype=GRB.BINARY, name="served")
    is_truck = model.addVars(customers, vtype=GRB.BINARY, name="is_truck")
    is_drone = model.addVars(customers, vtype=GRB.BINARY, name="is_drone")

    # --------------------------
    # 2. 基础路由与流平衡约束 (不变)
    # --------------------------
    for j in customers:
        model.addConstr(is_truck[j] == gp.quicksum(x[i, j, k] for i in nodes for k in range(num_trucks) if i != j))
        model.addConstr(is_drone[j] == gp.quicksum(y[i_, j_, h_, m_] for (i_, j_, h_, m_) in y_keys if j_ == j))
        model.addConstr(served[j] == is_truck[j] + is_drone[j])
        model.addConstr(served[j] <= 1)

    for k in range(num_trucks):
        model.addConstr(gp.quicksum(x[depot, j, k] for j in customers) <= 1)
        model.addConstr(gp.quicksum(x[i, depot, k] for i in customers) <= 1)
        for j in customers:
            model.addConstr(gp.quicksum(x[i, j, k] for i in nodes if i != j) == 
                            gp.quicksum(x[j, i, k] for i in nodes if i != j))

    for m in range(num_drones):
        model.addConstr(gp.quicksum(y[i, j, h, m_] for (i, j, h, m_) in y_keys if m_ == m) <= 1)

    # --------------------------
    # 3. 协同逻辑与路损约束 (不变)
    # --------------------------
    for i in nodes:
        for m in range(num_drones):
            model.addConstr(gp.quicksum(launch_by[i, k, m] for k in range(num_trucks)) == 
                            gp.quicksum(y[i, j, h, m] for j in customers for h in nodes if (i, j, h, m) in y_keys))
            for k in range(num_trucks):
                if i != depot:
                    model.addConstr(launch_by[i, k, m] <= gp.quicksum(x[j, i, k] for j in nodes if j != i))

    for h in nodes:
        for m in range(num_drones):
            model.addConstr(gp.quicksum(recv_by[h, k, m] for k in range(num_trucks)) == 
                            gp.quicksum(y[i, j, h, m] for i in nodes for j in customers if (i, j, h, m) in y_keys))
            for k in range(num_trucks):
                if h != depot:
                    model.addConstr(recv_by[h, k, m] <= gp.quicksum(x[j, h, k] for j in nodes if j != h))

    for i in customers:
        if is_damaged[i]:
            model.addConstr(is_truck[i] == 0)

    # --------------------------
    # 4. 载重约束 (Peng Eq. 45-46)
    # --------------------------
    for k in range(num_trucks):
        model.addConstr(gp.quicksum(demands[j] * x[i, j, k] for i in nodes for j in customers if i != j) <= Q_T)
    for (i, j, h, m) in y_keys:
        model.addConstr(y[i, j, h, m] * demands[j] <= Q_D)

    # --------------------------
    # 5. 时间与延迟约束 (核心灾难：复刻 Peng Eq. 54-63)
    # --------------------------
    bigM = horizon * 10
    
    arr_t = model.addVars(N, num_trucks, vtype=GRB.CONTINUOUS, name="arr_t")
    a_D   = model.addVars(N, num_drones, vtype=GRB.CONTINUOUS, name="a_D")
    
    # 🚨 修复 1：删除原有的 arr_t[depot, k] == 0 和 a_D[depot, m] == 0。
    # 我们让这两个变量专门表示“返回仓库的时间”。起飞时间 0 将直接在下方写死。

    # 5.1 卡车行程时间
    for k in range(num_trucks):
        for i in nodes:
            for j in nodes:
                if i != j:
                    if i == depot:
                        # 从仓库出发，起始时间直接视为 0
                        model.addConstr(arr_t[j, k] >= 0 + time_T[depot,j] - bigM*(1 - x[depot, j, k]))
                    else:
                        model.addConstr(arr_t[j, k] >= arr_t[i, k] + time_T[i,j] - bigM*(1 - x[i, j, k]))

    # 5.2 无人机飞行至客户节点 j 的时间
    for (i, j, h, m) in y_keys:
        for k1 in range(num_trucks):
            if i == depot:
                # 从仓库起飞，起始时间直接视为 0
                model.addConstr(a_D[j, m] >= 0 + time_D[depot,j] - bigM * (2 - y[depot,j,h,m] - launch_by[depot,k1,m]))
            else:
                # 从卡车起飞，起始时间为卡车到达时间
                model.addConstr(a_D[j, m] >= arr_t[i, k1] + time_D[i,j] - bigM * (2 - y[i,j,h,m] - launch_by[i,k1,m]))

    # 5.3 无人机飞行至汇合节点 h 的时间 (保持不变)
    for (i, j, h, m) in y_keys:
        for k2 in range(num_trucks):
            model.addConstr(a_D[h, m] >= a_D[j, m] + time_D[j,h] - 
                            bigM * (2 - y[i,j,h,m] - recv_by[h,k2,m]))

    # 5.4 起飞节点的严格时间同步
    for i in nodes:
        if i == depot:
            continue # 🚨 修复 2：仓库起飞时间默认是0，不需要和代表“返回时间”的 arr_t[depot] 同步
        for m in range(num_drones):
            for k1 in range(num_trucks):
                model.addConstr(a_D[i, m] >= arr_t[i, k1] - bigM * (1 - launch_by[i, k1, m]))
                model.addConstr(a_D[i, m] <= arr_t[i, k1] + bigM * (1 - launch_by[i, k1, m]))

    # 5.5 汇合节点的严格时间同步
    for h in nodes:
        if h == depot:
            continue # 🚨 修复 3：无人机降落在仓库时，独立回收，不需要强制和某辆卡车的返回时间分毫不差
        for m in range(num_drones):
            for k2 in range(num_trucks):
                model.addConstr(a_D[h, m] >= arr_t[h, k2] - bigM * (1 - recv_by[h, k2, m]))
                model.addConstr(a_D[h, m] <= arr_t[h, k2] + bigM * (1 - recv_by[h, k2, m]))

    # 5.6 使用大M法进行无人机电量限制 (保持不变)
    for (i, j, h, m) in y_keys:
        model.addConstr(time_D[i,j] + time_D[j,h] - bigM * (1 - y[i, j, h, m]) <= max_charge)

    # 5.7 延迟时间惩罚计算 (保持不变)
    delay_t = model.addVars(customers, vtype=GRB.CONTINUOUS, name="delay_t")
    for j in customers:
        model.addConstr(delay_t[j] >= 0)
        for k in range(num_trucks):
            model.addConstr(delay_t[j] >= arr_t[j, k] - deadlines[j] - bigM*(1 - gp.quicksum(x[i,j,k] for i in nodes if i!=j)))
        for (i, j_, h, m) in y_keys:
            if j_ == j:
                model.addConstr(delay_t[j] >= a_D[j, m] - deadlines[j] - bigM*(1 - y[i,j,h,m]))

    # # --------------------------
    # # 5. 时间与延迟约束 (核心灾难：复刻 Peng Eq. 54-63)
    # # --------------------------
    # bigM = horizon * 10
    
    # # 定义卡车和无人机的时间到达变量 (分别对应论文的 a_i^k 和 a'_i^k)
    # arr_t = model.addVars(N, num_trucks, vtype=GRB.CONTINUOUS, name="arr_t")
    # a_D   = model.addVars(N, num_drones, vtype=GRB.CONTINUOUS, name="a_D")
    
    # # 起点时间初始化 (Peng Eq. 51-52)
    # for k in range(num_trucks):
    #     model.addConstr(arr_t[depot, k] == 0)
    # for m in range(num_drones):
    #     model.addConstr(a_D[depot, m] == 0)

    # # 5.1 卡车行程时间 (Peng Eq. 54)
    # for k in range(num_trucks):
    #     for i in nodes:
    #         for j in nodes:
    #             if i != j:
    #                 model.addConstr(arr_t[j, k] >= arr_t[i, k] + time_T[i,j] - bigM*(1 - x[i, j, k]))

    # # 5.2 无人机飞行至客户节点 j 的时间 (Peng Eq. 55)
    # for (i, j, h, m) in y_keys:
    #     for k1 in range(num_trucks):
    #         # 无人机到达客户 j 的时间 >= 卡车到达发射点 i 的时间 + 飞行时间
    #         model.addConstr(a_D[j, m] >= arr_t[i, k1] + time_D[i,j] - 
    #                         bigM * (2 - y[i,j,h,m] - launch_by[i,k1,m]))

    # # 5.3 无人机飞行至汇合节点 h 的时间 (Peng Eq. 56)
    # for (i, j, h, m) in y_keys:
    #     for k2 in range(num_trucks):
    #         # 无人机到达汇合点 h 的时间 >= 无人机到达 j 的时间 + 飞行时间
    #         model.addConstr(a_D[h, m] >= a_D[j, m] + time_D[j,h] - 
    #                         bigM * (2 - y[i,j,h,m] - recv_by[h,k2,m]))

    # # 5.4 起飞节点的严格时间同步 (Peng Eq. 57, 58)
    # for i in nodes:
    #     for m in range(num_drones):
    #         for k1 in range(num_trucks):
    #             model.addConstr(a_D[i, m] >= arr_t[i, k1] - bigM * (1 - launch_by[i, k1, m]))
    #             model.addConstr(a_D[i, m] <= arr_t[i, k1] + bigM * (1 - launch_by[i, k1, m]))

    # # 5.5 汇合节点的严格时间同步 (Peng Eq. 59, 60)
    # for h in nodes:
    #     for m in range(num_drones):
    #         for k2 in range(num_trucks):
    #             model.addConstr(a_D[h, m] >= arr_t[h, k2] - bigM * (1 - recv_by[h, k2, m]))
    #             model.addConstr(a_D[h, m] <= arr_t[h, k2] + bigM * (1 - recv_by[h, k2, m]))

    # # 【灾难修改 2】5.6 使用大M法进行无人机电量限制 (Peng Eq. 61)
    # for (i, j, h, m) in y_keys:
    #     model.addConstr(time_D[i,j] + time_D[j,h] - bigM * (1 - y[i, j, h, m]) <= max_charge)

    # # 5.7 延迟时间惩罚计算 (Peng Eq. 63)
    # delay_t = model.addVars(customers, vtype=GRB.CONTINUOUS, name="delay_t")
    # for j in customers:
    #     model.addConstr(delay_t[j] >= 0)
    #     # 如果卡车访问，且超出 horizon
    #     for k in range(num_trucks):
    #         model.addConstr(delay_t[j] >= arr_t[j, k] - deadlines[j] - bigM*(1 - gp.quicksum(x[i,j,k] for i in nodes if i!=j)))
    #     # 🌟 修复：将 horizon 替换为 deadlines[j]
    #     for (i, j_, h, m) in y_keys:
    #         if j_ == j:
    #             model.addConstr(delay_t[j] >= a_D[j, m] - deadlines[j] - bigM*(1 - y[i,j,h,m]))
    #     # for k in range(num_trucks):
    #     #     model.addConstr(delay_t[j] >= arr_t[j, k] - horizon - bigM*(1 - gp.quicksum(x[i,j,k] for i in nodes if i!=j)))
    #     # # 如果无人机访问，且超出 horizon
    #     # for (i, j_, h, m) in y_keys:
    #     #     if j_ == j:
    #     #         model.addConstr(delay_t[j] >= a_D[j, m] - horizon - bigM*(1 - y[i,j,h,m]))

    # --------------------------
    # 6. 目标函数 (不变)
    # --------------------------
    cost_T = gp.quicksum(c_t * time_T[i,j] * x[i, j, k] for i in nodes for j in nodes if i!=j for k in range(num_trucks))
    cost_D = gp.quicksum(c_d * (time_D[i,j] + time_D[j,h]) * y[i, j, h, m] for (i, j, h, m) in y_keys)
    penalty_delay = gp.quicksum(alpha_penalty * delay_t[j] for j in customers)
    profit_service = gp.quicksum(reward_val * served[j] for j in customers)

    model.setObjective(profit_service - (cost_T + cost_D + penalty_delay), GRB.MAXIMIZE)

    # --------------------------
    # 求解配置与运行
    # --------------------------
    model.setParam('TimeLimit', 7200)
    model.setParam('MIPGap', 0.01)
    model.setParam('NodefileStart', 0.5)
    model.setParam('LogFile', 'gurobi_peng_style.log')

    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
        val = model.ObjVal
        status_msg = "最优解" if model.status == GRB.OPTIMAL else "次优解(超时)"
        print(f"✅ Gurobi 成功找到{status_msg}！目标值(Profit): {val:.2f} (耗时: {solve_time:.2f}s)")
        return val, solve_time, True
    else:
        print(f"❌ 未找到可行解 (状态码: {model.status}) (耗时: {solve_time:.2f}s)")
        return None, solve_time, False

# ===== 测试用例与运行入口 =====
if __name__ == "__main__":
    from trucks_and_drones.config import cfg
    import os
    
    test_json = "rl_instance_coords.json"
    
    if os.path.exists(test_json):
        config = cfg['environment']
        with open(test_json, 'r') as f:
            N = len(json.load(f)['coords_km'])
            
        demands = [0] + [np.random.randint(10, 30) for _ in range(N-1)]
        is_damaged = [False] * N
        for idx in np.random.choice(range(1, N), size=int(N * config['road_damage_ratio']), replace=False):
            is_damaged[idx] = True
            
        solve_flexible_vrpd_upper_bound(test_json, config, demands, is_damaged)

# import numpy as np
# import gurobipy as gp
# from gurobipy import GRB
# import math
# import json
# import time

# def solve_flexible_vrpd_upper_bound(json_path, config, demands, is_damaged):
#     """
#     求解【灵活跨车协同】(Flexible Cross-Truck Rendezvous) 的多卡车-多无人机 MIP 模型
#     完美契合 MACA-MA2C 算法中“无人机可返回任意卡车”的核心设定
#     """
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     coords = np.array(data['coords_km'])
#     N = len(coords)
#     nodes = list(range(N))
#     customers = list(range(1, N))
#     depot = 0
    
#     # 获取参数 (卡车和无人机数量不再强制绑定！)
#     num_trucks = config['num_trucks']
#     num_drones = config['num_drones']
#     v_t, v_d = config['truck_speed'], config['drone_speed']
#     c_t, c_d = config['ct_cost'], config['cd_cost']
#     Q_T, Q_D = config['WT_max'], config['WD_max']
#     max_charge = config['max_charge']
#     horizon = config['horizon']
#     reward_val = 30.0 
    
#     # 距离与时间矩阵
#     time_T = np.zeros((N, N))
#     time_D = np.zeros((N, N))
#     for i in nodes:
#         for j in nodes:
#             dist_M = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
#             dist_E = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
#             time_T[i, j] = dist_M / v_t
#             time_D[i, j] = dist_E / v_d

#     print(f"🚀 构建【灵活跨车协同】Gurobi 模型: {N-1}客户, {num_trucks}卡车, {num_drones}无人机")
    
#     model = gp.Model("Flexible_Truck_Drone_Routing")
#     model.setParam('TimeLimit', 7200)  
#     model.setParam('MIPGap', 0.01)     

#     # ================= 1. 核心决策变量 =================
    
#     # 卡车路径: 卡车 k 从 i 到 j
#     x = model.addVars(nodes, nodes, range(num_trucks), vtype=GRB.BINARY, name="x")
    
#     # 无人机飞行路径: 无人机 m 从 i 起飞, 服务 j, 在 h 降落
#     y_keys = [(i, j, h, m) for i in nodes for j in customers for h in nodes for m in range(num_drones)
#           if i != j and j != h and i != h]
#     y = model.addVars(y_keys, vtype=GRB.BINARY, name="y")
    
#     # 【核心解耦变量】：跨车起降分配
#     takeoff = model.addVars(nodes, range(num_drones), range(num_trucks), vtype=GRB.BINARY, name="takeoff")
#     land = model.addVars(nodes, range(num_drones), range(num_trucks), vtype=GRB.BINARY, name="land")
    
#     # 节点服务状态
#     z = model.addVars(customers, vtype=GRB.BINARY, name="z")
    
#     # 时间变量 (卡车到达节点时间)
#     arr_t = model.addVars(nodes, range(num_trucks), vtype=GRB.CONTINUOUS, lb=0, ub=horizon, name="arr_t")
#     # 载荷变量
#     load_T = model.addVars(nodes, range(num_trucks), vtype=GRB.CONTINUOUS, lb=0, ub=Q_T, name="load_T")

#     # ================= 2. 目标函数 =================
#     total_reward = gp.quicksum(reward_val * z[i] for i in customers)
#     total_truck_cost = gp.quicksum(c_t * time_T[i,j] * x[i,j,k] for i in nodes for j in nodes for k in range(num_trucks))
#     total_drone_cost = gp.quicksum(c_d * (time_D[i,j] + time_D[j,h]) * y[i,j,h,m] for (i,j,h,m) in y_keys)
    
#     model.setObjective(total_reward - total_truck_cost - total_drone_cost, GRB.MAXIMIZE)

#     # ================= 3. 约束条件 =================

#     # 3.1 节点服务约束 (同 Peng Eq.3-5)
#     for j in customers:
#         served_by_truck = gp.quicksum(x[i, j, k] for i in nodes for k in range(num_trucks) if i != j)
#         served_by_drone = gp.quicksum(y[i, j, h, m] for (i, j_, h, m) in y_keys if j_ == j)
#         model.addConstr(served_by_truck + served_by_drone == z[j])

#     # 3.2 卡车流平衡约束 (同 Peng Eq.6-11)
#     for k in range(num_trucks):
#         model.addConstr(gp.quicksum(x[depot, j, k] for j in customers) <= 1)
#         model.addConstr(gp.quicksum(x[i, depot, k] for i in customers) <= 1)
#         for j in customers:
#             model.addConstr(gp.quicksum(x[i, j, k] for i in nodes if i != j) == 
#                             gp.quicksum(x[j, out, k] for out in nodes if out != j))

#     # 3.3 道路受损约束 (环境特有，Peng 无此约束，RL必须有)
#     for i in customers:
#         if is_damaged[i]:
#             for j in nodes:
#                 for k in range(num_trucks):
#                     model.addConstr(x[i, j, k] == 0)
#                     model.addConstr(x[j, i, k] == 0)

#     # 3.4 无人机载重约束 (同 Peng Eq.55)
#     for (i, j, h, m) in y_keys:
#         model.addConstr(y[i, j, h, m] * demands[j] <= Q_D)

#     # ===== 新增：3.5 无人机电量约束 =====
#     for (i, j, h, m) in y_keys:
#         model.addConstr(y[i, j, h, m] * (time_D[i,j] + time_D[j,h]) <= max_charge)

#     # ---------------- 🚨 灵活跨车协同的核心约束 ----------------

#     # 3.5 绑定 y 与 takeoff/land
#     for m in range(num_drones):
#         for i in nodes:
#             # 如果无人机 m 从 i 起飞，它必须分配给恰好 1 辆停在 i 的卡车
#             model.addConstr(gp.quicksum(y[i, j, h, m] for j in customers for h in nodes if (i,j,h,m) in y_keys) == 
#                             gp.quicksum(takeoff[i, m, k] for k in range(num_trucks)))
#         for h in nodes:
#             # 如果无人机 m 在 h 降落，它必须分配给恰好 1 辆停在 h 的卡车
#             model.addConstr(gp.quicksum(y[i, j, h, m] for i in nodes for j in customers if (i,j,h,m) in y_keys) == 
#                             gp.quicksum(land[h, m, k] for k in range(num_trucks)))

#     # 3.6 空间同步：起降点必须有卡车 (Space Synchronization)
#     for m in range(num_drones):
#         for k in range(num_trucks):
#             for i in customers: # 不限制 depot
#                 model.addConstr(takeoff[i, m, k] <= gp.quicksum(x[in_node, i, k] for in_node in nodes if in_node != i))
#                 model.addConstr(land[i, m, k] <= gp.quicksum(x[in_node, i, k] for in_node in nodes if in_node != i))

#     # 3.7 跨车时间同步 (Cross-Truck Time Synchronization)
#     M_time = horizon + 100
#     for k in range(num_trucks):
#         for i in customers:
#             for j in customers:
#                 if i != j:
#                     # 卡车自身行驶时间推移
#                     model.addConstr(arr_t[j, k] >= arr_t[i, k] + time_T[i, j] - M_time * (1 - x[i, j, k]))
                    
#     # 【最华丽的约束】：如果无人机从卡车 k1 起飞，降落到卡车 k2，卡车 k2 的时间必须晚于卡车 k1 的时间 + 飞行时间
#     for (i, j, h, m) in y_keys:
#         if i != depot and h != depot:
#             for k1 in range(num_trucks):
#                 for k2 in range(num_trucks):
#                     # 如果 y=1 且 takeoff(k1)=1 且 land(k2)=1，则触发此约束 (利用大 M 法进行逻辑 AND)
#                     model.addConstr(arr_t[h, k2] >= arr_t[i, k1] + (time_D[i,j] + time_D[j,h]) - 
#                                     M_time * (3 - y[i,j,h,m] - takeoff[i,m,k1] - land[h,m,k2]))

#     # 3.8 卡车容量推移 (MTZ 消除子回路)
#     for k in range(num_trucks):
#         for i in customers:
#             for j in customers:
#                 if i != j:
#                     model.addConstr(load_T[j, k] >= load_T[i, k] + demands[j] - Q_T * (1 - x[i, j, k]))

#     # === 优化参数设置 ===
#     # 允许 Gurobi 使用硬盘空间作为虚拟内存（当内存用到 0.5 GB 时开始往硬盘写节点数据）
#     model.setParam('NodefileStart', 0.5) 
    
#     # 开始求解
#     start_time = time.time()
    
#     try:
#         model.optimize()
#     except gp.GurobiError as e:
#         # 捕获内存溢出(OOM)或其他致命错误，防止整个评估脚本崩溃
#         solve_time = time.time() - start_time
#         if e.errno == 10001:
#             print(f"💥 Gurobi 内存溢出 (OOM 10001)! 耗时: {solve_time:.2f}s")
#         else:
#             print(f"❌ Gurobi 发生严重错误: {e}")
#         return None, solve_time, False

#     solve_time = time.time() - start_time

#     if model.status == GRB.OPTIMAL:
#         print(f"✅ Gurobi 完美收敛 (最优解)! 用时: {solve_time:.2f} 秒")
#         print(f"💰 离线理论最优总利润 (Offline Optimum): {model.ObjVal:.2f}")
#         return model.ObjVal, solve_time, True
        
#     elif model.status == GRB.TIME_LIMIT:
#         try:
#             val = model.ObjVal
#             print(f"⚠️ Gurobi 达到时间限制 ({solve_time:.2f}s)! 返回当前次优解: {val:.2f}")
#             return val, solve_time, False
#         except AttributeError:
#             print(f"❌ Gurobi 超时 ({solve_time:.2f}s) 且未找到任何可行解!")
#             return None, solve_time, False
            
#     else:
#         print(f"❌ 未找到可行解 (状态码: {model.status})")
#         return None, solve_time, False
    
#     # # 开始求解
#     # start_time = time.time()
#     # model.optimize()
#     # solve_time = time.time() - start_time

#     # # if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
#     # #     print(f"✅ Gurobi 求解完成! 用时: {solve_time:.2f} 秒")
#     # #     print(f"💰 离线理论最优总利润 (Offline Optimum): {model.ObjVal:.2f}")
#     # #     return model.ObjVal, solve_time
#     # # else:
#     # #     print("❌ 未找到可行解")
#     # #     return None, solve_time
#     # if model.status == GRB.OPTIMAL:
#     #     print(f"✅ Gurobi 完美收敛 (最优解)! 用时: {solve_time:.2f} 秒")
#     #     print(f"💰 离线理论最优总利润 (Offline Optimum): {model.ObjVal:.2f}")
#     #     return model.ObjVal, solve_time, True
        
#     # elif model.status == GRB.TIME_LIMIT:
#     #     try:
#     #         val = model.ObjVal
#     #         print(f"⚠️ Gurobi 达到时间限制 ({solve_time:.2f}s)! 返回当前次优解: {val:.2f}")
#     #         return val, solve_time, False
#     #     except AttributeError:
#     #         print(f"❌ Gurobi 超时 ({solve_time:.2f}s) 且未找到任何可行解!")
#     #         return None, solve_time, False
            
#     # else:
#     #     print(f"❌ 未找到可行解 (状态码: {model.status})")
#     #     return None, solve_time, False

# # ===== 测试用例与运行入口 =====
# if __name__ == "__main__":
#     from trucks_and_drones.config import cfg
#     import os
    
#     # 假设你导出了 10 个节点的 json 到根目录
#     test_json = "rl_instance_coords.json"
    
#     if os.path.exists(test_json):
#         # 读取配置
#         config = cfg['environment']
        
#         # 伪造 demand 和 道路损坏数据 (你应该在 temp_database.py 导出时将这些写入 json)
#         with open(test_json, 'r') as f:
#             N = len(json.load(f)['coords_km'])
            
#         demands = [0] + [np.random.randint(10, 30) for _ in range(N-1)] # Depot 需求为0
#         is_damaged = [False] * N
#         # 模拟 20% 道路受损
#         damaged_indices = np.random.choice(range(1, N), size=int(N * 0.2), replace=False)
#         for idx in damaged_indices:
#             is_damaged[idx] = True
            
#         print(f"受损节点列表: {damaged_indices}")
        
#         # 运行 MIP
#         solve_flexible_vrpd_upper_bound(test_json, config, demands, is_damaged)
#     else:
#         print(f"未找到 {test_json}，请先在你的强化学习环境中运行 export_instance_coords()")