import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 路径配置 (使用您指定的绝对路径)
# ==========================================
FILE_PATH = r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\flexible\evaluation_MAPPO.csv"
# 默认输出到与表格相同的文件夹
OUT_DIR = os.path.dirname(FILE_PATH) 

# ==========================================
# 2. 绘图风格全局设置 (参考您的 plot_flexible.py)
# ==========================================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 18,
    'axes.labelsize': 18, 
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'axes.grid': False,
    'lines.linewidth': 2
})

# 颜色定义
COLOR_FLEX = '#376eae'   # 您的 Flexible 颜色
# COLOR_UNFLEX = '#376eae' # 对比组颜色
COLOR_UNFLEX = '#ef8a47' # 对比组颜色

def main():
    # ==========================================
    # 3. 数据处理
    # ==========================================
    if not os.path.exists(FILE_PATH):
        print(f"❌ 错误: 找不到文件 {FILE_PATH}")
        return

    # 读取 CSV (由于有两行表头，我们先读取全部数据)
    df = pd.read_csv(FILE_PATH, header=None)
    
    # 根据您的表格结构：最后一行的第一列是 'average'
    # 我们直接取最后一行的数据 (Index 14)
    avg_row = df[df[0] == 'average'].iloc[0]
    
    # 提取六个关键值 (列索引 2,3,4 为 Flex; 5,6,7 为 Unflex)
    flex_values = [
        float(avg_row[2]),  # Total Cost
        float(avg_row[3]),  # Total Episode time
        float(avg_row[4])   # delay_penalty
    ]
    
    unflex_values = [
        float(avg_row[5]),  # Total Cost
        float(avg_row[6]),  # Total Episode time
        float(avg_row[7])   # delay_penalty
    ]

    labels = ['Total Cost', 'Episode Time', 'Delay Penalty']
    
    # ==========================================
    # 4. 绘图逻辑 (双 Y 轴处理量级差异)
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(9, 7))

    x = np.arange(len(labels))
    width = 0.35

    # --- 左侧 Y 轴 (Cost 和 Time) ---
    rects1 = ax1.bar(x - width/2, flex_values, width, label='Flexible', 
                     color=COLOR_FLEX)
    rects2 = ax1.bar(x + width/2, unflex_values, width, label='Inflexible', 
                     color=COLOR_UNFLEX)

    # rects1 = ax1.bar(x - width/2, flex_values, width, label='Flexible', 
    #                  color=COLOR_FLEX, edgecolor='black', linewidth=0.8)
    # rects2 = ax1.bar(x + width/2, unflex_values, width, label='Unflexible', 
    #                  color=COLOR_UNFLEX, edgecolor='black', linewidth=0.8)

    ax1.set_ylabel('Cost / Time', fontsize=20)
    ax1.set_ylim(0, max(flex_values[0:2] + unflex_values[0:2]) * 1.2)
    
    # --- 右侧 Y 轴 (Delay Penalty) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Delay Penalty', fontsize=20)
    ax2.set_ylim(0, max(flex_values[2], unflex_values[2]) * 1.4)

    # 重点：在右轴上覆盖绘制第三组柱子，使其高度匹配右侧刻度
    ax2.bar(x[2] - width/2, flex_values[2], width, color=COLOR_FLEX)
    ax2.bar(x[2] + width/2, unflex_values[2], width, color=COLOR_UNFLEX)

    # ==========================================
    # 5. 细节修饰
    # ==========================================
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight='bold')
    
    # 数值标注函数
    def add_labels(rects, ax_target, indices):
        for i in indices:
            rect = rects[i]
            height = rect.get_height()
            ax_target.text(rect.get_x() + rect.get_width()/2., height + (height*0.02),
                           f'{height:.2f}', ha='center', va='bottom', fontsize=18)

    add_labels(rects1, ax1, [0, 1]) # Flex 前两个
    add_labels(rects2, ax1, [0, 1]) # Unflex 前两个
    
    # 单独标注第三组 (Penalty) 在右轴上的高度
    ax2.text(x[2] - width/2, flex_values[2] + 0.2, f'{flex_values[2]:.2f}', ha='center', va='bottom', fontsize=18)
    ax2.text(x[2] + width/2, unflex_values[2] + 0.2, f'{unflex_values[2]:.2f}', ha='center', va='bottom', fontsize=18)

    # 图例与分隔线
    ax1.legend(loc='upper right', frameon=True)
    ax1.axvline(x=1.5, color='gray', linestyle='--', alpha=0.3) # 分隔线区分量级
    
    # plt.title('Comparison of Average Performance Metrics', pad=20)
    plt.tight_layout()
    
    # 保存图片
    output_name = os.path.join(OUT_DIR, 'flexible_comparison_summary.png')
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已成功保存至: {output_name}")
    plt.show()

if __name__ == "__main__":
    main()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # ==========================================
# # 1. 配置路径
# # ==========================================
# FILE_PATH = r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\flexible\evaluation.csv"
# OUT_DIR = r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\flexible"

# # ==========================================
# # 2. 绘图风格全局设置
# # ==========================================
# plt.rcParams.update({
#     'font.family': 'Times New Roman',
#     'font.size': 16, # 全局基础字体
#     'axes.labelsize': 24, # 轴标签
#     'axes.titlesize': 18, # 标题
#     'xtick.labelsize': 14,  # X轴刻度
#     'ytick.labelsize': 14,  # Y轴刻度
#     'legend.fontsize': 14,  # 图例
#     'axes.grid': False,
#     'lines.linewidth': 2
# })

# # ==========================================
# # 3. 颜色定义 (根据您提供的RGB值)
# # ==========================================
# # 将 RGB (0-255) 转换为 Matplotlib 接受的归一化 RGB (0-1) 或者直接用 Hex
# # R:75, G:166, B:178 -> #4BA6B2
# COLOR_FLEX = '#719AAC'  

# # R:252, G:140, B:90 -> #FC8C5A
# COLOR_UNFLEX = '#FC8C5A' 

# def load_data(path):
#     """
#     读取并清洗数据
#     """
#     if not os.path.exists(path):
#         print(f"❌ 错误：找不到文件 - {path}")
#         return None

#     try:
#         df = pd.read_csv(path, header=None)
#         data = df.iloc[2:].copy()
        
#         # 重命名列
#         data.columns = [
#             'eta', 'rho', 
#             'flex_cost', 'flex_time', 'flex_penalty',
#             'unflex_cost', 'unflex_time', 'unflex_penalty'
#         ] + list(data.columns[8:])

#         # 填充 eta
#         data['eta'] = data['eta'].ffill()
        
#         # 删除 rho 为空的行
#         data = data.dropna(subset=['rho'])

#         # 转换为数值
#         cols = ['eta', 'rho', 'flex_cost', 'flex_time', 'flex_penalty', 
#                 'unflex_cost', 'unflex_time', 'unflex_penalty']
#         for col in cols:
#             data[col] = pd.to_numeric(data[col], errors='coerce')

#         # 删除无效数据行
#         data = data.dropna(subset=['flex_cost', 'unflex_cost'])

#         return data[cols]
#     except Exception as e:
#         print(f"❌ 数据处理异常: {e}")
#         return None

# def add_labels(ax, rects):
#     """
#     辅助函数：给柱状图添加数值标签（保留2位小数）
#     """
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height:.2f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 垂直偏移 3 个点
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=13, fontweight='bold')

# def plot_single_chart(subset, metric_flex, metric_unflex, rho_val, ylabel, filename):
#     """
#     【模式一】绘制单张图表 (9张小图)
#     """
#     labels = subset['eta'].unique()
#     labels.sort()
#     subset = subset.sort_values(by='eta')
    
#     flex_vals = subset[metric_flex].values
#     unflex_vals = subset[metric_unflex].values

#     x = np.arange(len(labels))
#     width = 0.35

#     # 保持较高的宽高比 (7, 6.5)
#     fig, ax = plt.subplots(figsize=(7, 6.5))
    
#     rects1 = ax.bar(x - width/2, flex_vals, width, label='Flexible', 
#                     color=COLOR_FLEX, edgecolor='black', alpha=1.0) # alpha=1.0 颜色更实
#     rects2 = ax.bar(x + width/2, unflex_vals, width, label='Unflexible', 
#                     color=COLOR_UNFLEX, edgecolor='black', alpha=1.0)

#     add_labels(ax, rects1)
#     add_labels(ax, rects2)

#     ax.set_xlabel(r'$\eta$')
#     ax.set_ylabel(ylabel)
#     ax.set_title(r'Comparison at $\rho_{dyn}=' + str(rho_val) + r'$')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.legend()
    
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     plt.tight_layout()
    
#     save_path = os.path.join(OUT_DIR, filename)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"  └─ [单图] 保存: {filename}")
#     plt.close()

# def plot_combined_metric(df, metric_config, target_rhos):
#     """
#     【模式二】绘制组合图表 (3张大图，每张含3个子图)
#     """
#     metric_name, flex_col, unflex_col, y_label, file_prefix = metric_config
    
#     # 保持较高的宽高比 (20, 8)
#     fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
#     if not isinstance(axes, np.ndarray):
#         axes = [axes]

#     for idx, (rho, ax) in enumerate(zip(target_rhos, axes)):
#         subset = df[df['rho'] == rho].sort_values(by='eta')
        
#         if subset.empty:
#             continue

#         labels = subset['eta'].unique()
#         flex_vals = subset[flex_col].values
#         unflex_vals = subset[unflex_col].values
        
#         x = np.arange(len(labels))
#         width = 0.35

#         rects1 = ax.bar(x - width/2, flex_vals, width, label='Flexible', 
#                         color=COLOR_FLEX, edgecolor='black', alpha=1.0)
#         rects2 = ax.bar(x + width/2, unflex_vals, width, label='Unflexible', 
#                         color=COLOR_UNFLEX, edgecolor='black', alpha=1.0)

#         add_labels(ax, rects1)
#         add_labels(ax, rects2)

#         ax.set_xlabel(r'$\eta$')
#         if idx == 0:
#             ax.set_ylabel(y_label)
        
#         ax.set_title(r'$\rho_{dyn} = ' + str(rho) + r'$')
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels)
        
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
        
#         if idx == 0:
#             ax.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False)

#     plt.tight_layout()
    
#     filename = f"Combined_{file_prefix}.png"
#     save_path = os.path.join(OUT_DIR, filename)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"  └─ [组合图] 保存: {filename}")
#     plt.close()

# def main():
#     if not os.path.exists(OUT_DIR):
#         try:
#             os.makedirs(OUT_DIR)
#         except OSError:
#             pass

#     # 1. 加载数据
#     df = load_data(FILE_PATH)
#     if df is None or df.empty:
#         print("❌ 数据加载失败或数据为空")
#         return

#     target_rhos = [0.2, 0.5, 0.7]
#     metrics_configs = [
#         ('Cost', 'flex_cost', 'unflex_cost', 'Total Cost', 'Cost'),
#         ('Time', 'flex_time', 'unflex_time', 'Total Episode Time (min)', 'Time'),
#         ('Penalty', 'flex_penalty', 'unflex_penalty', 'Delay Penalty', 'Penalty')
#     ]

#     print(f"🚀 开始绘图任务...")

#     # ==========================================
#     # 任务一：生成 9 张单图
#     # ==========================================
#     print(f"\n>> 正在生成 9 张独立小图")
#     for metric_name, flex_col, unflex_col, y_label, file_prefix in metrics_configs:
#         for rho in target_rhos:
#             subset = df[df['rho'] == rho]
#             if subset.empty: continue
            
#             filename = f"{file_prefix}_Rho{rho}.png"
#             plot_single_chart(subset, flex_col, unflex_col, rho, y_label, filename)

#     # ==========================================
#     # 任务二：生成 3 张组合大图
#     # ==========================================
#     print(f"\n>> 正在生成 3 张组合大图 (颜色：新蓝/新橙)...")
#     for config in metrics_configs:
#         plot_combined_metric(df, config, target_rhos)

#     print(f"\n✅ 全部完成！所有图片已保存至: {OUT_DIR}")

# if __name__ == "__main__":
#     main()