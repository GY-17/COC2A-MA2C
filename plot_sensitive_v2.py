import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse  # 新增：用于解析命令行参数
from mpl_toolkits.axes_grid1 import make_axes_locatable # 【新增】用于严格控制色条高度

# 设置全局字体大小，确保论文中清晰可见
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
    'font.family': 'sans-serif' # 显式设置字体族以防警告
})


def plot_dual_axis(df, x_col, y1_col, y2_col, xlabel, y1_label, y2_label, filename):
    """
    绘制双Y轴敏感性分析图的通用函数 (修复了X轴和Y轴标签被边框遮挡的问题)
    """
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # --- 左 Y 轴 (Total Cost) ---
    color1 = '#1f77b4'  # 经典蓝
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1_label, color=color1, fontweight='bold')
    
    # 绘制左轴线条
    line1 = ax1.plot(df[x_col], df[y1_col], color=color1, linestyle='-', marker='o', 
                     linewidth=2, markersize=8, label=y1_label)
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)

    # 添加左轴数值标签
    for x, y in zip(df[x_col], df[y1_col]):
        ax1.annotate(f'{y:.2f}', 
                     xy=(x, y), 
                     xytext=(8, 5),      # 向右上方偏移
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=15, 
                     color=color1)

    # 【新增】自动调整左 Y 轴范围：顶部预留 15% 空间
    y1_min, y1_max = df[y1_col].min(), df[y1_col].max()
    y1_margin = (y1_max - y1_min) * 0.15 if y1_max != y1_min else y1_max * 0.15
    ax1.set_ylim(bottom=y1_min - y1_margin * 0.5, top=y1_max + y1_margin)

    # --- 右 Y 轴 (Service Rate 或 Usage Ratio) ---
    ax2 = ax1.twinx()
    # color2 = '#ff7f0e'  # 经典橙
    color2 = '#48C0AA'  # 经典橙
    ax2.set_ylabel(y2_label, color=color2, fontweight='bold')
    
    # 绘制右轴线条
    line2 = ax2.plot(df[x_col], df[y2_col], color=color2, linestyle='--', marker='^', 
                     linewidth=2, markersize=8, label=y2_label)
    
    ax2.tick_params(axis='y', labelcolor=color2)

    # 添加右轴数值标签
    for x, y in zip(df[x_col], df[y2_col]):
        ax2.annotate(f'{y:.2f}', 
                     xy=(x, y), 
                     xytext=(6, -6),     # 向右下方偏移
                     textcoords="offset points",
                     ha='center', va='top',
                     fontsize=15, 
                     color=color2)

    # 【新增】自动调整右 Y 轴范围：底部预留 15% 空间
    y2_min, y2_max = df[y2_col].min(), df[y2_col].max()
    y2_margin = (y2_max - y2_min) * 0.15 if y2_max != y2_min else y2_max * 0.15
    ax2.set_ylim(bottom=y2_min - y2_margin, top=y2_max + y2_margin * 0.5)

    # ==============================================================================
    # 【新增】自动调整 X 轴范围：左右各预留 10% 空间，防止最右侧标签被压
    # ==============================================================================
    x_min, x_max = df[x_col].min(), df[x_col].max()
    x_margin = (x_max - x_min) * 0.10 if x_max != x_min else x_max * 0.10
    # 设置 X 轴范围 (left, right)
    ax1.set_xlim(left=x_min - x_margin * 0.5, right=x_max + x_margin)


    # --- 合并图例 ---
    lines = line1 + line2
    labels = [l.get_label() for l in lines]

    ax1.legend(lines, labels, 
               loc='upper right', 
               bbox_to_anchor=(0.97, 0.57), # 图例位置
               ncol=1, 
               frameon=True, 
               shadow=True)

    plt.tight_layout()
    
    print(f"Saving {filename}.png ...")
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    plt.close()

def plot_heatmap(df, x_col, y_col, value_col, xlabel, ylabel, cbar_label, filename):
    """
    绘制热力图 (优化版：使用 make_axes_locatable 强制对齐色条高度)
    """
    # 【修改1】移除 constrained_layout=True，防止与 make_axes_locatable 冲突
    # 我们将在保存时依靠 bbox_inches='tight' 来处理留白
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. 数据透视
    pivot_table = df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc='mean')
    
    # 【修改2】创建专门的 colorbar 轴 (cax)，使其高度严格匹配主图 ax
    divider = make_axes_locatable(ax)
    # append_axes("right", ...) 表示在右侧添加
    # size="5%" 控制色条宽度
    # pad=0.1 控制色条与热力图主体的间距 (可以根据需要微调，如 0.15)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # 2. 绘制热力图
    # 【修改3】cbar_kws 移除了 fraction/pad，因为由 cax 接管了布局
    sns.heatmap(pivot_table, 
                annot=True,       
                fmt=".2f",        
                # cmap="YlOrRd",   
                cmap="YlGnBu", 
                square=False,      # 保持正方形格子
                linewidths=.5,
                ax=ax,            # 主图画在 ax
                cbar_ax=cax,      # 【关键】指定色条画在 cax 上
                annot_kws={"size": 15, "weight": "normal"}, 
                cbar_kws={'label': cbar_label} 
                )

    # 3. 标签设置
    ax.set_xlabel(xlabel, fontsize=14, labelpad=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, labelpad=10, fontweight='bold')
    
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 获取 colorbar 对象并调整其标签字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12) # 色条刻度字体
    cbar.set_label(cbar_label, size=14, weight='bold') # 色条标题字体
    
    # 让Y轴刻度正常显示
    ax.invert_yaxis() 
    
    # 保存图片
    print(f"Saving {filename}.png ...")
    # 依靠 bbox_inches='tight' 自动裁剪多余留白
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    plt.close()

def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Draw Sensitivity Analysis Plots")
    
    # 允许用户指定数据目录，默认为你代码中的路径
    default_dir = r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\sensitive_analyze1"
    parser.add_argument('--dir', type=str, default=default_dir, 
                        help='Directory containing the result CSV files')
    
    # 允许用户指定绘制哪种类型
    parser.add_argument('--type', type=str, nargs='+', default=['all'],
                        choices=['capacity', 'battery', 'cost', 'speed', 'damage_dod', 'all'],
                        help='Which sensitivity analysis to plot: capacity, battery, cost, speed, damage_dod, or all')

    args = parser.parse_args()
    save_dir = args.dir
    
    print(f"📂 Working directory: {save_dir}")
    print(f"🎨 Plotting types: {args.type}")

    # ==========================================
    # 1. Drone Capacity (保持原样: Cost vs Service Rate)
    # ==========================================
    if 'all' in args.type or 'capacity' in args.type:
        file_path = os.path.join(save_dir, "drone_capacity_results.csv")
        if os.path.exists(file_path):
            print(f"✅ Processing: {file_path}")
            df = pd.read_csv(file_path)
            plot_dual_axis(
                df=df,
                x_col='capacity',
                y1_col='total_cost',
                y2_col='service_rate',
                xlabel=r'$W_{max}^{DR}$ [kg]',
                y1_label='Average Total Cost',
                y2_label='Service Rate',
                filename=os.path.join(save_dir, 'sensitivity_drone_capacity')
            )
        else:
            print(f"❌ Missing: {file_path}")

    # ==========================================
    # 2. Battery Capacity (保持原样: Cost vs Service Rate)
    # ==========================================
    if 'all' in args.type or 'battery' in args.type:
        file_path = os.path.join(save_dir, "battery_capacity_results.csv")
        if os.path.exists(file_path):
            print(f"✅ Processing: {file_path}")
            df = pd.read_csv(file_path)
            plot_dual_axis(
                df=df,
                x_col='battery',
                y1_col='total_cost',
                y2_col='service_rate',
                xlabel=r'$E_{max}$',
                y1_label='Average Total Cost',
                y2_label='Service Rate',
                filename=os.path.join(save_dir, 'sensitivity_battery_capacity')
            )
        else:
            print(f"❌ Missing: {file_path}")

    # ==========================================
    # 3. Cost Ratio (新增: Cost vs Drone Usage Ratio)
    # ==========================================
    if 'all' in args.type or 'cost' in args.type:
        file_path = os.path.join(save_dir, "cost_ratio_results.csv")
        if os.path.exists(file_path):
            print(f"✅ Processing: {file_path}")
            df = pd.read_csv(file_path)
            
            # 【核心逻辑】计算 Usage Ratio
            # 防止除零错误，加一个极小值
            df['usage_ratio'] = df['drone_travel_time'] / (df['truck_travel_time'] + df['drone_travel_time'] + 1e-6)
            
            plot_dual_axis(
                df=df,
                x_col='cost_ratio',
                y1_col='total_cost',
                y2_col='usage_ratio',  # 使用新计算的列
                xlabel=r'Cost Ratio ($c_d / c_t$)',
                y1_label='Average Total Cost',
                y2_label='Drone Usage Ratio',  # 图例和轴标签
                filename=os.path.join(save_dir, 'sensitivity_cost_ratio')
            )
        else:
            print(f"❌ Missing: {file_path}")

    # ==========================================
    # 4. Speed Ratio (新增: Cost vs Drone Usage Ratio)
    # ==========================================
    if 'all' in args.type or 'speed' in args.type:
        file_path = os.path.join(save_dir, "speed_ratio_results.csv")
        if os.path.exists(file_path):
            print(f"✅ Processing: {file_path}")
            df = pd.read_csv(file_path)
            
            # # 【核心逻辑】计算 Usage Ratio
            # df['usage_ratio'] = df['drone_travel_time'] / (df['truck_travel_time'] + df['drone_travel_time'] + 1e-6)
            
            plot_dual_axis(
                df=df,
                x_col='speed_ratio',
                y1_col='total_cost',
                y2_col='drone_wait_time',
                xlabel=r'$v_d / v_t$',
                y1_label='Average Total Cost',
                y2_label='Drone Wait Time',
                filename=os.path.join(save_dir, 'sensitivity_speed_ratio')
            )
        else:
            print(f"❌ Missing: {file_path}")

    # ==========================================
    # 5. Damage vs DoD (热力图)
    # ==========================================
    if 'all' in args.type or 'damage_dod' in args.type:
        file_path = os.path.join(save_dir, "damage_dod_results.csv")
        if os.path.exists(file_path):
            print(f"✅ Processing Heatmap: {file_path}")
            df = pd.read_csv(file_path)
            
            # 打印列名以供检查（调试用）
            print(f"   CSV Columns: {df.columns.tolist()}")

            # 1. 确定 X 轴 (受损度): 对应列名 '\eta'
            # 优先查找 '\eta'，如果没有则尝试其他可能
            if '\\eta' in df.columns:
                x_col = '\\eta'
            elif 'eta' in df.columns:
                x_col = 'eta'
            elif 'λ' in df.columns: # 兼容旧版
                x_col = 'λ'
            else:
                x_col = df.columns[1] # 最后的兜底：默认取第2列
                
            # 2. 确定 Y 轴 (动态度): 对应列名 '\rho_{dyn}'
            if '\\rho_{dyn}' in df.columns:
                y_col = '\\rho_{dyn}'
            elif 'rho_dyn' in df.columns:
                y_col = 'rho_dyn'
            elif 'DoD' in df.columns: # 兼容旧版
                y_col = 'DoD'
            else:
                y_col = df.columns[2] # 最后的兜底：默认取第3列

            print(f"   Using columns: X={x_col}, Y={y_col}, Value=total_cost")

            plot_heatmap(
                df=df,
                x_col=x_col, 
                y_col=y_col,
                value_col='total_cost',  # 绘制 Total Cost
                xlabel=r'Road Damage Ratio ($\eta$)', # X轴标签 (Latex格式)
                ylabel=r'Dynamic Request Degree ($\rho_{dyn}$)', # Y轴标签 (Latex格式)
                cbar_label='Average Total Cost',
                filename=os.path.join(save_dir, 'sensitivity_robustness_heatmap')
            )
        else:
            print(f"❌ Missing: {file_path}")

if __name__ == "__main__":
    main()