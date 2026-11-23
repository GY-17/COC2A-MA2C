import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime

from trucks_and_drones.config import cfg
from trucks_and_drones.build_env import BuildEnvironment
from evaluate_unifiedmaca2c_visualize import ModelEvaluator
from Gu import GuDynamicSolver

# 定义拦截器，屏蔽环境内部耗时的打印开销
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        # 🌟 关键修复：强制使用 utf-8 编码，防止 Windows 下 emoji 报错
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def run_gu_batch_experiments(args):
    env_cfg = cfg['environment']
    num_trucks = env_cfg['num_trucks']
    num_drones = env_cfg['num_drones']
    num_customers = env_cfg['num_customers']
    num_depots = env_cfg['num_depots']

    print(f"🚀 开始超级批量评估 (RL vs Gu et al. 2023)...")
    print(f"   - 实例数量: {args.instances} 个 (每个跑 {args.episodes} 回合)")
    print(f"   - 屏蔽了终端高频打印，以获取最纯净准确的底层推理耗时！\n")
    if args.skip_gu:
        print(f"   - ⚠️ 已启用 --skip_gu：仅评估 RL，Gu 启发式列（Gu_Profit/Gu_TravelCost/.../Gu_Time）将记为 NaN\n")

    evaluator = ModelEvaluator(
        checkpoint_dir=args.checkpoint_dir, 
        num_trucks=num_trucks, 
        num_drones=num_drones, 
        total_nodes=num_customers + num_depots, 
        algorithm=args.algorithm 
    )

    csv_data = []

    for damage in args.damages:
        for dod in args.dod_values:
            cfg['environment']['road_damage_ratio'] = damage
            cfg['environment']['dynamic_nodes']['dod'] = dod
            cfg['environment']['dynamic_nodes']['enable'] = True if dod > 0 else False

            end_index = args.start_index + args.instances
            for inst_idx in range(args.start_index, end_index):
                dataset_name = f"{args.prefix}{inst_idx}{args.suffix}"
                dataset_path = os.path.join("singlecenter", dataset_name) if os.path.exists(os.path.join("singlecenter", dataset_name)) else dataset_name
                cfg['environment']['dataset_path'] = dataset_path

                for ep in range(args.episodes):
                    builder = (
                        BuildEnvironment(name=f"Gu_Batch_D{damage}_DoD{dod}_Inst{inst_idx}_Ep{ep}", enable_visualization=args.visualize)
                        .trucks(num=num_trucks, speed=cfg['environment'].get('truck_speed'))
                        .drones(num=num_drones, speed=cfg['environment'].get('drone_speed'), 
                                max_cargo=cfg['environment'].get('WD_max'), max_charge=cfg['environment'].get('max_charge'))
                        .depots(num=num_depots)
                        .customers(num=num_customers)
                        .observations()
                        .actions()
                        .rewards()
                        .compile()
                    )
                    env = builder.build()
                    # 注意：你需要确保环境的 reset() 已经执行，通常在 build() 或者 evaluate 的开头
                    env.reset() # 强制初始化一下物理状态

                    # 屏蔽一切打印输出，获取极速真实推理时间
                    # with HiddenPrints():
                    rl_start_time = time.time()
                    rl_stats = evaluator.evaluate_single_episode_for_batch(env, render=args.visualize)
                    rl_time = time.time() - rl_start_time
                    
                    # 🌟 关键修复：让当前跑的这个随机环境把自己的地图导出给 Gu！
                    gu_json_path = f"temp_gu_instance_{inst_idx}_ep{ep}.json"

                    if args.skip_gu:
                        # --skip_gu：跳过 Gu 启发式，只跑 RL（用于快速验证 RL 或 Gu 不可用时）
                        gu_stats = None
                        gu_time = float('nan')
                    else:
                        env.temp_db.export_dynamic_instance_for_gu(path=gu_json_path)
                        gu_start_time = time.time()
                        try:
                            solver = GuDynamicSolver(gu_json_path)
                            gu_stats = solver.solve()
                        except Exception as e:
                            # 🌟 修复 2: 把错误打印出来，不要静默失败
                            print(f"\n[Error] Gu Solver Failed on Inst {inst_idx} Ep {ep}: {e}")
                            gu_stats = None
                        gu_time = time.time() - gu_start_time

                    if gu_stats and isinstance(gu_stats, dict):
                        gu_profit = gu_stats.get('real_reward', 0)
                        gu_travel = gu_stats.get('travel_cost', 0)
                        gu_delay = gu_stats.get('delay_penalty', 0)
                        gu_unserved = gu_stats.get('unserved_penalty', 0)
                        gu_served_cnt = gu_stats.get('served_count', 0)
                    elif args.skip_gu:
                        # 显式跳过：Gu 列记 NaN，避免用 0 误导
                        gu_profit = gu_travel = gu_delay = gu_unserved = gu_served_cnt = float('nan')
                    else:
                        gu_profit = gu_travel = gu_delay = gu_unserved = gu_served_cnt = 0

                    rl_profit = rl_stats.get('real_reward', 0)
                    rl_travel = rl_stats.get('travel_cost', 0)
                    rl_delay = rl_stats.get('delay_penalty', 0)
                    rl_unserved = rl_stats.get('unserved_penalty', 0)
                    rl_served_cnt = rl_stats.get('served_count', 0)
                    
                    gap_profit = ((rl_profit - gu_profit) / (abs(gu_profit) + 1e-6)) * 100 if gu_profit != 0 else 0.0

                    # 细化记录指标，方便在论文表格中对比优势到底在哪里
                    csv_data.append({
                        'Damage_Ratio': damage,
                        'DoD': dod,
                        'RL_Served': rl_served_cnt,
                        'Gu_Served': gu_served_cnt,
                        'RL_Profit': rl_profit,
                        'Gu_Profit': gu_profit,
                        'RL_TravelCost': rl_travel,
                        'Gu_TravelCost': gu_travel,
                        'RL_DelayPenalty': rl_delay,
                        'Gu_DelayPenalty': gu_delay,
                        'RL_DropPenalty': rl_unserved,
                        'Gu_DropPenalty': gu_unserved,
                        'RL_Time(s)': rl_time,
                        'Gu_Time(s)': gu_time,
                        'Profit_Gap(%)': gap_profit
                    })
                    
                    if args.visualize and hasattr(env, 'visualizer'):
                        env.visualizer.close()

    # ==========================================================
    # 数据聚合：直接打印并保存每一组 (Damage, DoD) 的平均值
    # ==========================================================
    if csv_data:
        df = pd.DataFrame(csv_data)
        summary_df = df.groupby(['Damage_Ratio', 'DoD']).mean().reset_index()
        
        print("\n" + "="*110)
        print("📊 最终对比汇总结果 (按组合取平均值)")
        print("="*110)
        print(summary_df.to_string(index=False, float_format="%.2f"))
        print("="*110)
        
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(args.save_dir, f"Gu_vs_RL_nodes{num_customers}_FULL_{timestamp}.csv")
        summary_path = os.path.join(args.save_dir, f"Gu_vs_RL_nodes{num_customers}_SUMMARY_{timestamp}.csv")
        
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 数据已保存至:\n  明细数据: {full_path}\n  平均汇总: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量验证 RL 与 Gu(2023) 的对比实验")
    parser.add_argument("--start_index", type=int, default=51, help="数据集序号起始值")
    parser.add_argument("--instances", type=int, default=10, help="要测试的数据集数量")
    parser.add_argument("--episodes", type=int, default=3, help="每个数据集跑的回合数")
    parser.add_argument("--prefix", type=str, default="singlecenter-")
    parser.add_argument("--suffix", type=str, default="-n10.txt")
    parser.add_argument("--algorithm", type=str, default="maca2c")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="模型权重目录")
    parser.add_argument("--save_dir", type=str, default="./benchmark_results/Gu_vs_RL", help="CSV保存目录")
    parser.add_argument("--visualize", action="store_true", help="是否展示可视化界面")
    parser.add_argument("--damages", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--dod_values", type=float, nargs="+", default=[0.2, 0.5, 0.7])
    parser.add_argument("--skip_gu", action="store_true", help="跳过 Gu 启发式求解，仅运行 RL 评估（用于快速验证 RL 或 Gu 不可用时），Gu 列记为 NaN")
    
    args = parser.parse_args()
    run_gu_batch_experiments(args)

# import os
# import sys
# import time
# import argparse
# import pandas as pd
# from datetime import datetime

# from trucks_and_drones.config import cfg
# from trucks_and_drones.build_env import BuildEnvironment
# from evaluate_unifiedmaca2c_visualize import ModelEvaluator
# from Gu import GuDynamicSolver

# # 定义拦截器，屏蔽环境内部耗时的打印开销
# class HiddenPrints:
#     def __enter__(self):
#         self._original_stdout = sys.stdout
#         # 🌟 关键修复：强制使用 utf-8 编码，防止 Windows 下 emoji 报错
#         sys.stdout = open(os.devnull, 'w', encoding='utf-8')
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._original_stdout

# def run_gu_batch_experiments(args):
#     env_cfg = cfg['environment']
#     num_trucks = env_cfg['num_trucks']
#     num_drones = env_cfg['num_drones']
#     num_customers = env_cfg['num_customers']
#     num_depots = env_cfg['num_depots']

#     print(f"🚀 开始超级批量评估 (RL vs Gu et al. 2023)...")
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
#                         BuildEnvironment(name=f"Gu_Batch_D{damage}_DoD{dod}_Inst{inst_idx}_Ep{ep}", enable_visualization=args.visualize)
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
                        
#                         gu_json_path = "dynamic_instance.json"
                        
#                         gu_start_time = time.time()
#                         try:
#                             solver = GuDynamicSolver(gu_json_path)
#                             gu_stats = solver.solve()
#                         except Exception as e:
#                             gu_stats = None
#                         gu_time = time.time() - gu_start_time

#                     if gu_stats and isinstance(gu_stats, dict):
#                         gu_profit = gu_stats.get('real_reward', 0)
#                         gu_travel = gu_stats.get('travel_cost', 0)
#                         gu_delay = gu_stats.get('delay_penalty', 0)
#                         gu_unserved = gu_stats.get('unserved_penalty', 0)
#                         gu_served_cnt = gu_stats.get('served_count', 0)
#                     else:
#                         gu_profit = gu_travel = gu_delay = gu_unserved = gu_served_cnt = 0

#                     rl_profit = rl_stats.get('real_reward', 0)
#                     rl_travel = rl_stats.get('travel_cost', 0)
#                     rl_delay = rl_stats.get('delay_penalty', 0)
#                     rl_unserved = rl_stats.get('unserved_penalty', 0)
#                     rl_served_cnt = rl_stats.get('served_count', 0)
                    
#                     gap_profit = ((rl_profit - gu_profit) / (abs(gu_profit) + 1e-6)) * 100 if gu_profit != 0 else 0.0

#                     # 细化记录指标，方便在论文表格中对比优势到底在哪里
#                     csv_data.append({
#                         'Damage_Ratio': damage,
#                         'DoD': dod,
#                         'RL_Served': rl_served_cnt,
#                         'Gu_Served': gu_served_cnt,
#                         'RL_Profit': rl_profit,
#                         'Gu_Profit': gu_profit,
#                         'RL_TravelCost': rl_travel,
#                         'Gu_TravelCost': gu_travel,
#                         'RL_DelayPenalty': rl_delay,
#                         'Gu_DelayPenalty': gu_delay,
#                         'RL_DropPenalty': rl_unserved,
#                         'Gu_DropPenalty': gu_unserved,
#                         'RL_Time(s)': rl_time,
#                         'Gu_Time(s)': gu_time,
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
#         full_path = os.path.join(args.save_dir, f"Gu_vs_RL_nodes{num_customers}_FULL_{timestamp}.csv")
#         summary_path = os.path.join(args.save_dir, f"Gu_vs_RL_nodes{num_customers}_SUMMARY_{timestamp}.csv")
        
#         df.to_csv(full_path, index=False, encoding='utf-8-sig')
#         summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
#         print(f"\n💾 数据已保存至:\n  明细数据: {full_path}\n  平均汇总: {summary_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="批量验证 RL 与 Gu(2023) 的对比实验")
#     parser.add_argument("--start_index", type=int, default=51, help="数据集序号起始值")
#     parser.add_argument("--instances", type=int, default=10, help="要测试的数据集数量")
#     parser.add_argument("--episodes", type=int, default=3, help="每个数据集跑的回合数")
#     parser.add_argument("--prefix", type=str, default="singlecenter-")
#     parser.add_argument("--suffix", type=str, default="-n10.txt")
#     parser.add_argument("--algorithm", type=str, default="maca2c")
#     parser.add_argument("--checkpoint_dir", type=str, required=True, help="模型权重目录")
#     parser.add_argument("--save_dir", type=str, default="./benchmark_results/Gu_vs_RL", help="CSV保存目录")
#     parser.add_argument("--visualize", action="store_true", help="是否展示可视化界面")
#     parser.add_argument("--damages", type=float, nargs="+", default=[0.1, 0.3, 0.5])
#     parser.add_argument("--dod_values", type=float, nargs="+", default=[0.2, 0.5, 0.7])
    
#     args = parser.parse_args()
#     run_gu_batch_experiments(args)