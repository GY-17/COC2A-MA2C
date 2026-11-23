import os
import json
import numpy as np
import argparse
import pandas as pd
from datetime import datetime

from trucks_and_drones.config import cfg
from trucks_and_drones.build_env import BuildEnvironment
from MIP import solve_flexible_vrpd_upper_bound
from evaluate_unifiedmaca2c_visualize import ModelEvaluator

def run_batch_experiments(args):
    env_cfg = cfg['environment']
    num_trucks = env_cfg['num_trucks']
    num_drones = env_cfg['num_drones']
    num_customers = env_cfg['num_customers']
    num_depots = env_cfg['num_depots']

    print(f"🚀 开始超级批量评估...")
    print(f"   - 实例数量: {args.instances} 个数据集 (从序号 {args.start_index} 开始)")
    print(f"   - 测试回合: 每个实例 {args.episodes} 个 Episode")
    print(f"   - 算法模型: {args.algorithm}")
    print(f"   - 待测受损率: {args.damages}\n")
    
    evaluator = ModelEvaluator(
        checkpoint_dir=args.checkpoint_dir, 
        num_trucks=num_trucks, 
        num_drones=num_drones, 
        total_nodes=num_customers + num_depots, 
        algorithm=args.algorithm 
    )

    damage_summary_results = {}

    for damage in args.damages:
        print("\n" + "★"*60)
        print(f"🌪️ 开始测试道路受损率 (Damage Ratio) = {damage}")
        print("★"*60)
        
        env_cfg['road_damage_ratio'] = damage
        
        rl_profits, mip_profits, profit_gaps = [], [], []
        rl_costs, mip_costs = [], []
        rl_times, mip_times, time_speedups = [], [], []
        
        # 记录该 Damage 下 MIP 成功求解的次数
        mip_success_count = 0 
        total_runs = 0

        end_index = args.start_index + args.instances
        for i in range(args.start_index, end_index):
            dataset_name = f"{args.prefix}{i}{args.suffix}"
            dataset_path = os.path.join("singlecenter", dataset_name) if os.path.exists(os.path.join("singlecenter", dataset_name)) else dataset_name
            env_cfg['dataset_path'] = dataset_path
            # BuildEnvironment reads the shared cfg during compile/reset.
            cfg['environment']['dataset_path'] = dataset_path
            
            for ep in range(args.episodes):
                total_runs += 1
                print(f"\n--- [Damages: {damage}] 实例 {i} | 回合 {ep + 1}/{args.episodes} ---")
                
                builder = (
                    BuildEnvironment(name=f"BatchTest_D{damage}_Inst{i}_Ep{ep}", enable_visualization=args.visualize)
                    .trucks(num=num_trucks, speed=env_cfg.get('truck_speed'))
                    .drones(num=num_drones, speed=env_cfg.get('drone_speed'), 
                            max_cargo=env_cfg.get('WD_max'), max_charge=env_cfg.get('max_charge'))
                    .depots(num=num_depots)
                    .customers(num=num_customers)
                    .observations().actions().rewards().compile()
                )
                env = builder.build()
                
                if args.visualize and hasattr(env, 'visualizer'):
                    env.visualizer.enabled = True
                    env.visualizer.no_open_window = True
                
                print("🤖 RL 智能体开始推理...")
                stats = evaluator.evaluate_single_episode_for_batch(env, render=args.visualize)
                
                rl_profit = stats.get('real_reward', 0.0) 
                rl_cost = stats.get('total_cost', 0.0)
                rl_time = stats.get('inference_time', 0.001)

                print("🧮 Gurobi MIP 开始计算理论上限...")
                json_path = "rl_instance_coords.json"
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    demands = data['demands']
                    is_damaged = data['is_damaged']

                # mip_profit, mip_time = solve_flexible_vrpd_upper_bound(
                #     json_path=json_path, config=env_cfg, demands=demands, is_damaged=is_damaged
                # )

                # actual_customers = len(demands) - 1
                # total_service_reward = actual_customers * 30.0  

                # # ========================================================
                # # 💡 核心修复：就算超时算不出答案，时间必须强制吃满！
                # # 利润和 Gap 记作 np.nan (缺失值)，不影响有效均值的计算。
                # # ========================================================
                # if mip_profit is None:
                #     print(f"⚠️ MIP 求解失败/超时！强制记录死亡耗时: {mip_time:.2f}s")
                #     mip_profit_val = np.nan
                #     mip_cost_val = np.nan
                #     profit_gap = np.nan
                # else:
                #     mip_profit_val = mip_profit
                #     mip_cost_val = total_service_reward - mip_profit 
                #     profit_gap = ((mip_profit - rl_profit) / mip_profit) * 100 if mip_profit > 0 else 0.0
                #     mip_success_count += 1
                
                try:
                    mip_result = solve_flexible_vrpd_upper_bound(
                        json_path=json_path, config=env_cfg, demands=demands, is_damaged=is_damaged
                    )
                except Exception as e:
                    # Gurobi 许可证过期 / 未安装等环境错误：跳过 MIP，仅保留 RL 结果，不让整个 batch 崩溃
                    print(f"⚠️ MIP 求解器不可用（{type(e).__name__}: {e}），跳过本实例 MIP，仅记录 RL 结果")
                    mip_profit, mip_time, is_mip_optimal = None, 0.0, False
                else:
                    if len(mip_result) == 3:
                        mip_profit, mip_time, is_mip_optimal = mip_result
                    else:
                        mip_profit, mip_time = mip_result
                        is_mip_optimal = True if mip_profit is not None else False

                actual_customers = len(demands) - 1
                total_service_reward = actual_customers * 30.0  

                if mip_profit is None:
                    print(f"⚠️ MIP 求解失败/超时！强制记录死亡耗时: {mip_time:.2f}s")
                    mip_profit_val = np.nan
                    mip_cost_val = np.nan
                    profit_gap = np.nan
                else:
                    mip_profit_val = mip_profit
                    mip_cost_val = total_service_reward - mip_profit 
                    profit_gap = ((mip_profit - rl_profit) / mip_profit) * 100 if mip_profit > 0 else 0.0
                    
                    if is_mip_optimal:
                        mip_success_count += 1

                speedup = mip_time / rl_time

                if mip_profit is not None:
                    print(f"📊 战报: [利润] RL: {rl_profit:.2f} | MIP: {mip_profit:.2f} ➡️ Gap: {profit_gap:.2f}%")
                    print(f"         [成本] RL: {rl_cost:.2f} | MIP: {mip_cost_val:.2f}")
                print(f"         [时间] RL: {rl_time:.4f}s | MIP: {mip_time:.2f}s ➡️ 快了 {speedup:.0f} 倍")

                rl_profits.append(rl_profit)
                mip_profits.append(mip_profit_val)
                profit_gaps.append(profit_gap)
                
                rl_costs.append(rl_cost)
                mip_costs.append(mip_cost_val)
                
                rl_times.append(rl_time)
                mip_times.append(mip_time)
                time_speedups.append(speedup)

            if args.visualize and hasattr(env, 'visualizer'):
                env.visualizer.close()

        if total_runs > 0:
            # 💡 使用 np.nanmean 来忽略 NaN 值求利润均值，但是时间 time 是全员参与计算的！
            damage_summary_results[damage] = {
                'mip_profit': np.nanmean(mip_profits) if mip_success_count > 0 else np.nan, 
                'rl_profit': np.mean(rl_profits),
                'mip_cost': np.nanmean(mip_costs) if mip_success_count > 0 else np.nan,
                'rl_cost': np.mean(rl_costs),
                'mip_time': np.mean(mip_times),      # 所有局数的时间严格平摊，绝不放过任何超时！
                'rl_time': np.mean(rl_times),
                'gap': np.nanmean(profit_gaps) if mip_success_count > 0 else np.nan, 
                'speedup': np.mean(time_speedups),
                'success_rate': (mip_success_count / total_runs) * 100
            }

    # ================= 终极大表格输出 =================
    print("\n\n" + "="*140)
    print("🏆 终极实验报告：不同受损率 (Damages) 下的 MIP vs RL 综合性能对比 🏆")
    print("="*140)
    print(f"{'受损率':<8} | {'MIP成功率':<10} | {'MIP 利润':<10} | {'RL 利润':<10} | {'MIP 成本':<10} | {'RL 成本':<10} | {'MIP 耗时(s)':<12} | {'RL 耗时(s)':<11} | {'Profit Gap':<10} | {'加速比'}")
    print("-" * 140)
    
    csv_data = []
    for dmg in args.damages:
        if dmg in damage_summary_results:
            res = damage_summary_results[dmg]
            mip_profit_str = f"{res['mip_profit']:<12.2f}" if not np.isnan(res['mip_profit']) else f"{'超时无解':<12}"
            mip_cost_str = f"{res['mip_cost']:<12.2f}" if not np.isnan(res['mip_cost']) else f"{'超时无解':<12}"
            gap_str = f"{res['gap']:<10.2f}%" if not np.isnan(res['gap']) else f"{'N/A':<10}"
            
            print(f"{dmg:<10} | {res['success_rate']:<9.1f}% | {mip_profit_str} | {res['rl_profit']:<12.2f} | {mip_cost_str} | {res['rl_cost']:<12.2f} | {res['mip_time']:<14.2f} | {res['rl_time']:<13.4f} | {gap_str} | {res['speedup']:.0f}x")
            
            csv_data.append({
                "Damage Ratio": dmg, 
                "MIP Success Rate (%)": round(res['success_rate'], 1),
                "MIP Profit (Optimal)": round(res['mip_profit'], 2) if not np.isnan(res['mip_profit']) else "Timeout",
                "RL Profit (Model)": round(res['rl_profit'], 2), 
                "MIP Cost": round(res['mip_cost'], 2) if not np.isnan(res['mip_cost']) else "Timeout",
                "RL Cost": round(res['rl_cost'], 2),
                "MIP Time (s)": round(res['mip_time'], 4),
                "RL Time (s)": round(res['rl_time'], 4),
                "Optimality Gap (%)": round(res['gap'], 2) if not np.isnan(res['gap']) else "N/A",
                "Speedup (Times)": round(res['speedup'], 0)
            })
    print("="*140)

    if len(csv_data) > 0:
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"MIP_vs_RL_Comprehensive_nodes{num_customers}_{timestamp}.csv"
        save_path = os.path.join(args.save_dir, save_filename)
        df = pd.DataFrame(csv_data)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 成功！超全对比结果表格已保存至: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量验证 RL 与 MIP 对比实验")
    parser.add_argument("--start_index", type=int, default=1, help="数据集序号起始值")
    parser.add_argument("--visualize", action="store_true", help="是否在批量测试时展示可视化界面")
    parser.add_argument("--instances", type=int, default=10, help="要测试的数据集数量")
    parser.add_argument("--episodes", type=int, default=1, help="每个数据集跑的回合数")
    parser.add_argument("--prefix", type=str, default="singlecenter-")
    parser.add_argument("--suffix", type=str, default="-n5.txt")
    parser.add_argument("--algorithm", type=str, default="maca2c", help="使用的算法名称")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="模型权重目录")
    parser.add_argument("--damages", nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3], help="测试的多个受损率")
    parser.add_argument("--save_dir", type=str, default="./results", help="导出的 CSV 表格保存路径")
    args = parser.parse_args()
    run_batch_experiments(args)
