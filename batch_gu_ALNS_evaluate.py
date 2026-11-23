import os
import sys
import glob
import json
import time
import argparse
import pandas as pd
import warnings

# 导入配置和核心环境构建模块
from trucks_and_drones.config import cfg
from trucks_and_drones.build_env import BuildEnvironment

# 导入三种算法的评估入口
from evaluate_unifiedmaca2c_visualize import evaluate_rl
from Gu import CrossTruckGu_Rolling
from ALNS_roling import CrossTruckALNS_Rolling

warnings.filterwarnings('ignore')

class HiddenPrints:
    """上下文管理器：用于在批量评估时屏蔽算法内部的冗余 print 输出，保持控制台整洁"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def run_batch_experiments(args):
    # ================= 1. 获取测试实例 =================
    if args.instance:
        test_files = [args.instance]
    else:
        test_files = glob.glob(os.path.join(args.test_dir, "*.json"))
        if not test_files:
            print(f"❌ 在目录 '{args.test_dir}' 中没有找到测试文件！")
            return

    num_trucks = cfg['environment'].get('num_trucks', 5)
    num_drones = cfg['environment'].get('num_drones', 5)

    print(f"\n🚀 开始批量评估，共找到 {len(test_files)} 个测试实例...")
    print(f"⚙️ 当前环境配置: {num_trucks} 辆卡车, {num_drones} 架无人机\n")

    results = []

    for idx, instance_file in enumerate(test_files):
        print(f"[{idx+1}/{len(test_files)}] 正在评估实例: {os.path.basename(instance_file)}")
        
        try:
            with open(instance_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                num_nodes = len(data.get('nodes', []))
        except Exception as e:
            print(f"  ❌ 读取实例失败: {e}")
            continue

        instance_res = {
            'Instance': os.path.basename(instance_file),
            'Num_Nodes': num_nodes,
            'RL_Profit': 0.0, 'Gu_Profit': 0.0, 'ALNS_Profit': 0.0,
            'RL_Served(%)': 0.0, 'Gu_Served(%)': 0.0, 'ALNS_Served(%)': 0.0,
            'RL_T_Dist': 0.0, 'Gu_T_Dist': 0.0, 'ALNS_T_Dist': 0.0,
            'RL_D_Dist': 0.0, 'Gu_D_Dist': 0.0, 'ALNS_D_Dist': 0.0,
            'Gu_Gap(%)': 0.0, 'ALNS_Gap(%)': 0.0
        }

        # -----------------------------------------------------------------
        # 0. 核心修正：动态构建强化学习环境 (BuildEnvironment)
        # -----------------------------------------------------------------
        # 必须将当前的 json 实例路径塞入 config，然后才能 build_env
        cfg['environment']['dataset_path'] = instance_file
        try:
            # 实例化当前实例的真实物理环境，为 RL 准备
            env = BuildEnvironment(cfg)
        except Exception as e:
            print(f"  ❌ 环境 BuildEnvironment 初始化失败: {e}")
            continue

        # -----------------------------------------------------------------
        # 算法 1: MACA-MA2C (你的 RL 算法)
        # -----------------------------------------------------------------
        print(f"  -> 运行 MACA-MA2C (RL)...", end="", flush=True)
        start_time = time.time()
        try:
            # 将初始化好的 env 传给 evaluate_rl 函数
            if args.verbose:
                rl_stats = evaluate_rl(env, instance_file) 
            else:
                with HiddenPrints():
                    rl_stats = evaluate_rl(env, instance_file)
            
            instance_res['RL_Profit'] = rl_stats.get('real_reward', 0.0)
            instance_res['RL_Served(%)'] = len(rl_stats.get('served_nodes', [])) / max(1, num_nodes - 1) * 100
            instance_res['RL_T_Dist'] = rl_stats.get('truck_distance', 0.0)
            instance_res['RL_D_Dist'] = rl_stats.get('drone_distance', 0.0)
            print(f" ✅ 完成 ({time.time() - start_time:.2f}s) | 净利润: {instance_res['RL_Profit']:.2f}")
        except Exception as e:
            print(f" ❌ 失败! 错误信息: {e}")

        # -----------------------------------------------------------------
        # 算法 2: Gu Baseline (启发式)
        # -----------------------------------------------------------------
        print(f"  -> 运行 Gu Baseline...", end="", flush=True)
        start_time = time.time()
        try:
            # 注意：传统运筹学算子（Gu 和 ALNS）内部自己写了数据解析和距离矩阵计算，
            # 所以它们不需要传入 env 对象，依然传入 instance_file 即可
            if args.verbose:
                gu_solver = CrossTruckGu_Rolling(instance_file)
                gu_solver.solve()
                gu_stats = gu_solver.calculate_final_profit()
            else:
                with HiddenPrints():
                    gu_solver = CrossTruckGu_Rolling(instance_file)
                    gu_solver.solve()
                    gu_stats = gu_solver.calculate_final_profit()
                    
            instance_res['Gu_Profit'] = gu_stats.get('real_reward', 0.0)
            instance_res['Gu_Served(%)'] = len(gu_stats.get('served_nodes', [])) / max(1, num_nodes - 1) * 100
            instance_res['Gu_T_Dist'] = gu_stats.get('truck_distance', 0.0)
            instance_res['Gu_D_Dist'] = gu_stats.get('drone_distance', 0.0)
            print(f" ✅ 完成 ({time.time() - start_time:.2f}s) | 净利润: {instance_res['Gu_Profit']:.2f}")
        except Exception as e:
            print(f" ❌ 失败! 错误信息: {e}")

        # -----------------------------------------------------------------
        # 算法 3: ALNS-rolling Baseline (自适应大邻域搜索)
        # -----------------------------------------------------------------
        print(f"  -> 运行 ALNS-rolling Baseline...", end="", flush=True)
        start_time = time.time()
        try:
            if args.verbose:
                alns_solver = CrossTruckALNS_Rolling(instance_file)
                alns_solver.solve()
                alns_stats = alns_solver.calculate_final_profit()
            else:
                with HiddenPrints():
                    alns_solver = CrossTruckALNS_Rolling(instance_file)
                    alns_solver.solve()
                    alns_stats = alns_solver.calculate_final_profit()
                    
            instance_res['ALNS_Profit'] = alns_stats.get('real_reward', 0.0)
            instance_res['ALNS_Served(%)'] = len(alns_stats.get('served_nodes', [])) / max(1, num_nodes - 1) * 100
            instance_res['ALNS_T_Dist'] = alns_stats.get('truck_distance', 0.0)
            instance_res['ALNS_D_Dist'] = alns_stats.get('drone_distance', 0.0)
            print(f" ✅ 完成 ({time.time() - start_time:.2f}s) | 净利润: {instance_res['ALNS_Profit']:.2f}")
        except Exception as e:
            print(f" ❌ 失败! 错误信息: {e}")

        # -----------------------------------------------------------------
        # 4. 计算指标 Gap（以 RL_Profit 为基准）
        # -----------------------------------------------------------------
        rl_profit = instance_res['RL_Profit']
        if rl_profit != 0:
            instance_res['Gu_Gap(%)'] = (instance_res['Gu_Profit'] - rl_profit) / abs(rl_profit) * 100
            instance_res['ALNS_Gap(%)'] = (instance_res['ALNS_Profit'] - rl_profit) / abs(rl_profit) * 100

        results.append(instance_res)
        print("-" * 60)

    # ================= 3. 数据汇总与保存 =================
    if results:
        df = pd.DataFrame(results)
        output_csv = args.output
        df.to_csv(output_csv, index=False)
        
        print("\n" + "="*90)
        print("📊 FINAL BENCHMARK RESULTS (RL vs Gu vs ALNS)")
        print("="*90)
        print(df[['Instance', 'Num_Nodes', 'RL_Profit', 'Gu_Profit', 'ALNS_Profit', 'Gu_Gap(%)', 'ALNS_Gap(%)']].to_string(index=False))
        print("="*90)
        
        print("\n📈 性能均值对比 (Averages):")
        print(f"  [MACA-MA2C] | 均值利润: {df['RL_Profit'].mean():8.2f} | 均值服务率: {df['RL_Served(%)'].mean():5.1f}%")
        print(f"  [Gu]        | 均值利润: {df['Gu_Profit'].mean():8.2f} | 均值服务率: {df['Gu_Served(%)'].mean():5.1f}% | 相对RL差距: {df['Gu_Gap(%)'].mean():6.2f}%")
        print(f"  [ALNS]      | 均值利润: {df['ALNS_Profit'].mean():8.2f} | 均值服务率: {df['ALNS_Served(%)'].mean():5.1f}% | 相对RL差距: {df['ALNS_Gap(%)'].mean():6.2f}%")
        print("="*90)
        print(f"✅ 完整评估结果已保存至 -> {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL vs Gu vs ALNS baselines.")
    parser.add_argument("--test_dir", type=str, default="test_instances", help="包含测试 JSON 文件的目录")
    parser.add_argument("--instance", type=str, default="", help="指定单个实例文件进行测试 (可选)")
    parser.add_argument("--output", type=str, default="evaluation_results_gu_alns.csv", help="输出的 CSV 文件名")
    parser.add_argument("--verbose", action="store_true", help="打印求解器的详细日志 (不使用 HiddenPrints)")
    args = parser.parse_args()
    
    run_batch_experiments(args)

# import os
# import sys
# import time
# import argparse
# import pandas as pd
# from datetime import datetime

# from trucks_and_drones.config import cfg
# from trucks_and_drones.build_env import BuildEnvironment
# from evaluate_unifiedmaca2c_visualize import ModelEvaluator
# from ALNS_roling import CrossTruckALNS_Rolling

# # 定义拦截器，屏蔽环境内部耗时的打印开销
# class HiddenPrints:
#     def __enter__(self):
#         self._original_stdout = sys.stdout
#         # 强制使用 utf-8 编码，防止 Windows 下 emoji 报错
#         sys.stdout = open(os.devnull, 'w', encoding='utf-8')
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._original_stdout

# def run_alns_batch_experiments(args):
#     env_cfg = cfg['environment']
#     num_trucks = env_cfg['num_trucks']
#     num_drones = env_cfg['num_drones']
#     num_customers = env_cfg['num_customers']
#     num_depots = env_cfg['num_depots']

#     print(f"🚀 开始超级批量评估 (RL vs ALNS Baseline)...")
#     print(f"   - 实例数量: {args.instances} 个 (每个跑 {args.episodes} 回合)")
#     print(f"   - 屏蔽了终端高频打印，以获取最纯净准确的底层推理耗时！\n")

#     evaluator = ModelEvaluator(
#         checkpoint_dir=args.checkpoint_dir, 
#         num_trucks=num_trucks, 
#         num_drones=num_drones, 
#         total_nodes=num_customers + num_depots, 
#         algorithm=args.algorithm 
#     )

#     csv_data = []

#     for damage in args.damages:
#         for dod in args.dod_values:
#             cfg['environment']['road_damage_ratio'] = damage
#             cfg['environment']['dynamic_nodes']['dod'] = dod
#             cfg['environment']['dynamic_nodes']['enable'] = True if dod > 0 else False

#             end_index = args.start_index + args.instances
#             for inst_idx in range(args.start_index, end_index):
#                 dataset_name = f"{args.prefix}{inst_idx}{args.suffix}"
#                 dataset_path = os.path.join("singlecenter", dataset_name) if os.path.exists(os.path.join("singlecenter", dataset_name)) else dataset_name
#                 cfg['environment']['dataset_path'] = dataset_path

#                 for ep in range(args.episodes):
#                     builder = (
#                         BuildEnvironment(name=f"ALNS_Batch_D{damage}_DoD{dod}_Inst{inst_idx}_Ep{ep}", enable_visualization=args.visualize)
#                         .trucks(num=num_trucks, speed=cfg['environment'].get('truck_speed'))
#                         .drones(num=num_drones, speed=cfg['environment'].get('drone_speed'), 
#                                 max_cargo=cfg['environment'].get('WD_max'), max_charge=cfg['environment'].get('max_charge'))
#                         .depots(num=num_depots)
#                         .customers(num=num_customers)
#                         .observations()
#                         .actions()
#                         .rewards()
#                         .compile()
#                     )
#                     env = builder.build()

#                     # 屏蔽一切打印输出，获取极速真实推理时间
#                     with HiddenPrints():
#                         rl_start_time = time.time()
#                         rl_stats = evaluator.evaluate_single_episode_for_batch(env, render=args.visualize)
#                         rl_time = time.time() - rl_start_time
                        
#                         alns_json_path = "dynamic_instance.json"
                        
#                         alns_start_time = time.time()
#                         try:
#                             solver = CrossTruckALNS_Rolling(alns_json_path)
#                             solver.solve()
#                             alns_stats = solver.calculate_final_profit()
#                         except Exception as e:
#                             alns_stats = None
#                         alns_time = time.time() - alns_start_time

#                     # ==========================================
#                     # 提取 ALNS 统计数据
#                     # ==========================================
#                     if alns_stats and isinstance(alns_stats, dict):
#                         alns_profit = alns_stats.get('real_reward', 0)
#                         alns_travel = alns_stats.get('travel_cost', 0)
#                         alns_delay = alns_stats.get('delay_penalty', 0)
                        
#                         # ALNS_roling.py 将 served_nodes 存储为 set
#                         alns_served_nodes = alns_stats.get('served_nodes', set())
#                         alns_served_cnt = len(alns_served_nodes)
                        
#                         # 手动计算 unserved_penalty (ALNS_roling.py 内部计算了但未独立存入字典)
#                         unserved_count = num_customers - alns_served_cnt
#                         beta = cfg['node']['customer'].get('beta', 30.0)
#                         alns_unserved = unserved_count * beta
#                     else:
#                         alns_profit = alns_travel = alns_delay = alns_unserved = alns_served_cnt = 0

#                     # ==========================================
#                     # 提取 RL 统计数据
#                     # ==========================================
#                     rl_profit = rl_stats.get('real_reward', 0)
#                     rl_travel = rl_stats.get('travel_cost', 0)
#                     rl_delay = rl_stats.get('delay_penalty', 0)
#                     rl_unserved = rl_stats.get('unserved_penalty', 0)
#                     rl_served_cnt = rl_stats.get('served_count', 0)
                    
#                     gap_profit = ((rl_profit - alns_profit) / (abs(alns_profit) + 1e-6)) * 100 if alns_profit != 0 else 0.0

#                     # 细化记录指标
#                     csv_data.append({
#                         'Damage_Ratio': damage,
#                         'DoD': dod,
#                         'RL_Served': rl_served_cnt,
#                         'ALNS_Served': alns_served_cnt,
#                         'RL_Profit': rl_profit,
#                         'ALNS_Profit': alns_profit,
#                         'RL_TravelCost': rl_travel,
#                         'ALNS_TravelCost': alns_travel,
#                         'RL_DelayPenalty': rl_delay,
#                         'ALNS_DelayPenalty': alns_delay,
#                         'RL_DropPenalty': rl_unserved,
#                         'ALNS_DropPenalty': alns_unserved,
#                         'RL_Time(s)': rl_time,
#                         'ALNS_Time(s)': alns_time,
#                         'Profit_Gap(%)': gap_profit
#                     })
                    
#                     if args.visualize and hasattr(env, 'visualizer'):
#                         env.visualizer.close()

#     # ==========================================================
#     # 数据聚合：直接打印并保存每一组 (Damage, DoD) 的平均值
#     # ==========================================================
#     if csv_data:
#         df = pd.DataFrame(csv_data)
#         summary_df = df.groupby(['Damage_Ratio', 'DoD']).mean().reset_index()
        
#         print("\n" + "="*110)
#         print("📊 最终对比汇总结果 (按组合取平均值)")
#         print("="*110)
#         print(summary_df.to_string(index=False, float_format="%.2f"))
#         print("="*110)
        
#         os.makedirs(args.save_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         full_path = os.path.join(args.save_dir, f"ALNS_vs_RL_nodes{num_customers}_FULL_{timestamp}.csv")
#         summary_path = os.path.join(args.save_dir, f"ALNS_vs_RL_nodes{num_customers}_SUMMARY_{timestamp}.csv")
        
#         df.to_csv(full_path, index=False, encoding='utf-8-sig')
#         summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
#         print(f"\n💾 数据已保存至:\n  明细数据: {full_path}\n  平均汇总: {summary_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="批量验证 RL 与 ALNS 的对比实验")
#     parser.add_argument("--start_index", type=int, default=51, help="数据集序号起始值")
#     parser.add_argument("--instances", type=int, default=10, help="要测试的数据集数量")
#     parser.add_argument("--episodes", type=int, default=3, help="每个数据集跑的回合数")
#     parser.add_argument("--prefix", type=str, default="singlecenter-")
#     parser.add_argument("--suffix", type=str, default="-n10.txt")
#     parser.add_argument("--algorithm", type=str, default="maca2c")
#     parser.add_argument("--checkpoint_dir", type=str, required=True, help="模型权重目录")
#     parser.add_argument("--save_dir", type=str, default="./benchmark_results/ALNS_vs_RL", help="CSV保存目录")
#     parser.add_argument("--visualize", action="store_true", help="是否展示可视化界面")
#     parser.add_argument("--damages", type=float, nargs="+", default=[0.1, 0.3, 0.5])
#     parser.add_argument("--dod_values", type=float, nargs="+", default=[0.2, 0.5, 0.7])
    
#     args = parser.parse_args()
#     run_alns_batch_experiments(args)