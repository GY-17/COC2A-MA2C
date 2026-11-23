import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime

from trucks_and_drones.config import cfg
from trucks_and_drones.build_env import BuildEnvironment
from evaluate_unifiedmaca2c_visualize import ModelEvaluator
from ALNS_roling import CrossTruckALNS_Rolling

# 定义拦截器，屏蔽环境内部耗时的打印开销
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        # 强制使用 utf-8 编码，防止 Windows 下 emoji 报错
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def run_alns_batch_experiments(args):
    env_cfg = cfg['environment']
    num_trucks = env_cfg['num_trucks']
    num_drones = env_cfg['num_drones']
    num_customers = env_cfg['num_customers']
    num_depots = env_cfg['num_depots']

    print(f"🚀 开始超级批量评估 (RL vs ALNS Baseline)...")
    print(f"   - 实例数量: {args.instances} 个 (每个跑 {args.episodes} 回合)")
    print(f"   - 屏蔽了终端高频打印，以获取最纯净准确的底层推理耗时！\n")

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
                        BuildEnvironment(name=f"ALNS_Batch_D{damage}_DoD{dod}_Inst{inst_idx}_Ep{ep}", enable_visualization=args.visualize)
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

                    # # 屏蔽一切打印输出，获取极速真实推理时间
                    # with HiddenPrints():
                    rl_start_time = time.time()
                    rl_stats = evaluator.evaluate_single_episode_for_batch(env, render=args.visualize)
                    rl_time = time.time() - rl_start_time
                    
                    alns_json_path = "dynamic_instance.json"
                    
                    alns_start_time = time.time()
                    try:
                        solver = CrossTruckALNS_Rolling(alns_json_path)
                        solver.solve()
                        alns_stats = solver.calculate_final_profit()
                    except Exception as e:
                        alns_stats = None
                    alns_time = time.time() - alns_start_time

                    # ==========================================
                    # 提取 ALNS 统计数据
                    # ==========================================
                    if alns_stats and isinstance(alns_stats, dict):
                        alns_profit = alns_stats.get('real_reward', 0)
                        alns_travel = alns_stats.get('travel_cost', 0)
                        alns_delay = alns_stats.get('delay_penalty', 0)
                        
                        # ALNS_roling.py 将 served_nodes 存储为 set
                        alns_served_nodes = alns_stats.get('served_nodes', set())
                        alns_served_cnt = len(alns_served_nodes)
                        
                        # 手动计算 unserved_penalty (ALNS_roling.py 内部计算了但未独立存入字典)
                        unserved_count = num_customers - alns_served_cnt
                        beta = cfg['node']['customer'].get('beta', 30.0)
                        alns_unserved = unserved_count * beta
                    else:
                        alns_profit = alns_travel = alns_delay = alns_unserved = alns_served_cnt = 0

                    # ==========================================
                    # 提取 RL 统计数据
                    # ==========================================
                    rl_profit = rl_stats.get('real_reward', 0)
                    rl_travel = rl_stats.get('travel_cost', 0)
                    rl_delay = rl_stats.get('delay_penalty', 0)
                    rl_unserved = rl_stats.get('unserved_penalty', 0)
                    rl_served_cnt = rl_stats.get('served_count', 0)
                    
                    gap_profit = ((rl_profit - alns_profit) / (abs(alns_profit) + 1e-6)) * 100 if alns_profit != 0 else 0.0

                    # 细化记录指标
                    csv_data.append({
                        'Damage_Ratio': damage,
                        'DoD': dod,
                        'RL_Served': rl_served_cnt,
                        'ALNS_Served': alns_served_cnt,
                        'RL_Profit': rl_profit,
                        'ALNS_Profit': alns_profit,
                        'RL_TravelCost': rl_travel,
                        'ALNS_TravelCost': alns_travel,
                        'RL_DelayPenalty': rl_delay,
                        'ALNS_DelayPenalty': alns_delay,
                        'RL_DropPenalty': rl_unserved,
                        'ALNS_DropPenalty': alns_unserved,
                        'RL_Time(s)': rl_time,
                        'ALNS_Time(s)': alns_time,
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
        full_path = os.path.join(args.save_dir, f"ALNS_vs_RL_nodes{num_customers}_FULL_{timestamp}.csv")
        summary_path = os.path.join(args.save_dir, f"ALNS_vs_RL_nodes{num_customers}_SUMMARY_{timestamp}.csv")
        
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 数据已保存至:\n  明细数据: {full_path}\n  平均汇总: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量验证 RL 与 ALNS 的对比实验")
    parser.add_argument("--start_index", type=int, default=51, help="数据集序号起始值")
    parser.add_argument("--instances", type=int, default=10, help="要测试的数据集数量")
    parser.add_argument("--episodes", type=int, default=3, help="每个数据集跑的回合数")
    parser.add_argument("--prefix", type=str, default="singlecenter-")
    parser.add_argument("--suffix", type=str, default="-n10.txt")
    parser.add_argument("--algorithm", type=str, default="maca2c")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="模型权重目录")
    parser.add_argument("--save_dir", type=str, default="./benchmark_results/ALNS_vs_RL", help="CSV保存目录")
    parser.add_argument("--visualize", action="store_true", help="是否展示可视化界面")
    parser.add_argument("--damages", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--dod_values", type=float, nargs="+", default=[0.2, 0.5, 0.7])
    
    args = parser.parse_args()
    run_alns_batch_experiments(args)