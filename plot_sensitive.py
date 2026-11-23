import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置全局字体大小，确保论文中清晰可见
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

def plot_dual_axis(df, x_col, y1_col, y2_col, xlabel, y1_label, y2_label, filename):
    """
    绘制双Y轴敏感性分析图的通用函数
    
    参数:
    - df: DataFrame 数据源
    - x_col: X轴列名
    - y1_col: 左Y轴列名 (Cost)
    - y2_col: 右Y轴列名 (Service Rate)
    - xlabel: X轴标签
    - y1_label: 左Y轴标签
    - y2_label: 右Y轴标签
    - title: 图表标题
    - filename: 保存的文件名
    """
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # --- 左 Y 轴 (Total Cost) ---
    color1 = '#1f77b4'  # 经典蓝
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1_label, color=color1, fontweight='bold')
    
    # 绘制左轴线条: 实线 + 圆点
    line1 = ax1.plot(df[x_col], df[y1_col], color=color1, linestyle='-', marker='o', 
                     linewidth=2, markersize=8, label='Total Cost')
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)  # 只给左轴加网格

    # --- 为左轴 (Total Cost) 添加数值标签 ---
    # 遍历数据点，在点的上方 (xytext=(0, 8)) 显示数值，保留2位小数
    for x, y in zip(df[x_col], df[y1_col]):
        ax1.annotate(f'{y:.2f}', 
                     xy=(x, y), 
                     xytext=(2, 6),      # 偏移量：(x=0, y=8)
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=9, 
                     color=color1,
                     )

    # --- 右 Y 轴 (Service Rate) ---
    ax2 = ax1.twinx()  # 实例化第二个坐标轴，共享X轴
    color2 = '#ff7f0e'  # 经典橙
    ax2.set_ylabel(y2_label, color=color2, fontweight='bold')
    
    # 绘制右轴线条: 虚线 + 三角点
    line2 = ax2.plot(df[x_col], df[y2_col], color=color2, linestyle='--', marker='^', 
                     linewidth=2, markersize=8, label='Service Rate')
    
    ax2.tick_params(axis='y', labelcolor=color2)
    # 设置右轴范围，确保服务率显示美观 (可选，例如 0.8 到 1.05)
    # ax2.set_ylim(0.5, 1.05) 

    # 【新增】添加 Service Rate 数值标签 (下方)
    for x, y in zip(df[x_col], df[y2_col]):
        ax2.annotate(f'{y:.2f}', 
                     xy=(x, y), 
                     xytext=(0, -12),    # 偏移量：(x=0, y=-12)
                     textcoords="offset points",
                     ha='center', va='top',
                     fontsize=9, 
                     color=color2)

    # --- 合并图例 ---
    # 因为有两个轴，需要手动合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]

    # ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
    #            ncol=2, frameon=True, shadow=True)
    # 修改说明：
    # 1. loc='upper right': 基准锚点设为右上角
    # 2. bbox_to_anchor=(0.95, 0.90): 这是关键。(1, 1)是图框的最右上角。
    #    0.95 表示在 X 轴方向靠右留一点空隙；
    #    0.85 (或0.90) 表示在 Y 轴方向从顶部往下挪 10%-15%，正好避开顶部的服务率曲线。
    # 3. ncol=1: 改为单列垂直排列（即 total cost 在上，service rate 在下），
    #    这样比 ncol=2 更窄，不容易碰到左边下降的蓝色曲线。
    
    ax1.legend(lines, labels, 
               loc='upper right', 
               bbox_to_anchor=(0.96, 0.88), # 这里可以微调：第二个数字越小，图例越往下挪
               ncol=1, 
               frameon=True, 
               shadow=True)

    # plt.title(title, pad=40)
    plt.tight_layout()
    
    # 保存图片
    print(f"Saving {filename}.png ...")
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight') # 同时也存PDF供Latex使用
    plt.show()

def main():
    # 1. 无人机载重数据文件 (变量名已改为 capacity_file)
    capacity_file = r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\sensitive_analyze1\drone_capacity_results.csv"    
    # 2. 电池容量数据文件
    battery_file = r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\sensitive_analyze1\battery_capacity_results.csv"
    # 定义图片保存的文件夹路径
    save_dir = r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\sensitive_analyze1"
    # --- 调试检查 ---
    print(f"正在读取文件: {capacity_file}")

    # 1. 绘制 Drone Capacity 敏感性分析
    if os.path.exists(capacity_file):
        print(f"✅ 成功找到: {capacity_file}")
        df_drone = pd.read_csv(capacity_file)
        plot_dual_axis(
            df=df_drone,
            x_col='capacity',
            y1_col='total_cost',
            y2_col='service_rate',
            xlabel=r'Drone Max Capacity ($W_{max}^{DR}$) [kg]',  # 支持Latex公式
            y1_label='Total Cost',
            y2_label='Service Rate',
            # title='Sensitivity Analysis: Drone Capacity',
            filename=os.path.join(save_dir, 'sensitivity_drone_capacity')
        )
    else:
        print(f"❌ 找不到文件: {capacity_file}")

    # 2. 绘制 Battery Capacity 敏感性分析
    if os.path.exists(battery_file):
        print(f"✅ 成功找到: {battery_file}")
        df_batt = pd.read_csv(battery_file)
        plot_dual_axis(
            df=df_batt,
            x_col='battery',
            y1_col='total_cost',
            y2_col='service_rate',
            xlabel=r'Battery Capacity ($E_{max}$) [km]',
            y1_label='Total Cost',
            y2_label='Service Rate',
            # title='Sensitivity Analysis: Battery Capacity',
            filename=os.path.join(save_dir, 'sensitivity_battery_capacity')
        )
    else:
        print(f"❌ 找不到文件: {battery_file}")

if __name__ == "__main__":
    main()