import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.core.util import event_pb2

# ==================== 1. 配置区域 ====================
FILE_PATHS = {
    # 请填入您的真实路径
    # "Flexible":    r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\reward\MADDPGevents.out.tfevents.1765379728.autodl-container-c10245bd8d-3b243c4f.1539.1.v2",
    # "Inflexible": r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\flexible\pairs\episode\events.out.tfevents.1761277457.autodl-container-80d0408bc2-b1e288aa.2702.1.v2"
    "COC2A-MA2C(Transformer)": r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\reward\COC2Aevents.out.tfevents.1787757587.autodl-container-271149a41f-5415b5f0.26531.1.v2",   
    # "COCA(Transformer)": r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\reward\MACA-MA2Cevents.out.tfevents.filtered_2000.v2",   
    # "MADDPG":    r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\reward\MADDPGevents.out.tfevents.1765379728.autodl-container-c10245bd8d-3b243c4f.1539.1.v2",
    "COC2A-MA2C(MLP)": r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\reward\MLPevents.out.tfevents.1764158278.autodl-container-c10245bd8d-3b243c4f.15520.1.v2",
    # "MACA-MA2C": r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\reward\MACA-MA2Cevents.out.tfevents.filtered_2000.v2",
    "MADDPG":    r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\reward\MADDPGevents.out.tfevents.1765379728.autodl-container-c10245bd8d-3b243c4f.1539.1.v2",
    "MA2C":      r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\reward\MA2Cevents.out.tfevents.modified.1763522932.autodl-container-c10245bd8d-3b243c4f.3967.1.v2",
    "MAPPO":     r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\reward\MAPPOevents.out.tfevents.1765379699.autodl-container-d7fa43adb1-4548ccbc.3455.1.v2"
    

}

POSSIBLE_TAGS = ['Episode/Real_Reward', 'Episode/Total_Reward', 'real_reward', 'total_reward']

# --- 🏆 核心修改区：拉大差距的参数 🏆 ---

# 1. 强制裁剪 Y 轴范围 (Zoom In)
# 如果您的最终收敛值大概在 600 左右，这里设 [-200, 800] 效果最好。
# 千万不要设到 -10000，否则曲线又会粘在一起！
Y_LIMITS = [560, 700] 

# 2. 图片长宽比 (Stretch)
# (8, 10) 表示 宽10英寸，高8英寸。也就是一个“瘦高”的图，专门用来拉伸纵坐标。
FIGURE_SIZE = (10, 7) 

SMOOTH_WINDOW = 60   
MAX_STEPS = 1001    

# ==================== 2. 读取函数 (保持不变) ====================
def read_tfevents(algo_name, file_path):
    data = []
    print(f"🔍 分析 [{algo_name}]...", end=" ")
    if not os.path.exists(file_path): return pd.DataFrame()

    target_tag = None
    try:
        dataset = tf.data.TFRecordDataset(file_path)
        for i, raw_record in enumerate(dataset):
            if i > 500: break 
            event = event_pb2.Event.FromString(raw_record.numpy())
            if event.HasField('summary'):
                for value in event.summary.value:
                    if value.tag in POSSIBLE_TAGS:
                        if target_tag is None or (target_tag == 'Episode/Total_Reward' and value.tag == 'Episode/Real_Reward'):
                            target_tag = value.tag
        if not target_tag: return pd.DataFrame()

        dataset = tf.data.TFRecordDataset(file_path)
        for raw_record in dataset:
            event = event_pb2.Event.FromString(raw_record.numpy())
            if event.step > MAX_STEPS: continue
            if event.HasField('summary'):
                for value in event.summary.value:
                    if value.tag == target_tag:
                        val = None
                        if value.HasField('tensor'):
                            try: val = float(tf.make_ndarray(value.tensor))
                            except: pass
                        elif value.HasField('simple_value'):
                            val = value.simple_value
                        if val is not None:
                            data.append({'step': event.step, 'value': val})
    except: pass
    df = pd.DataFrame(data)
    if not df.empty: print(f"✅")
    else: print("⚠️")
    return df

# ==================== 3. 绘图主程序 ====================
def plot_zoomed_curves():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4) # 字体加大
    
    # 使用刚才设置的“瘦高”比例
    plt.figure(figsize=FIGURE_SIZE)
    
    colors = sns.color_palette("bright", n_colors=len(FILE_PATHS))
    
    has_plotted = False
    for i, (name, path) in enumerate(FILE_PATHS.items()):
        df = read_tfevents(name, path)
        if not df.empty:
            has_plotted = True
            # 平滑
            df['smoothed'] = df['value'].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
            df['std'] = df['value'].rolling(window=SMOOTH_WINDOW, min_periods=1).std().fillna(0)
            
            # 绘图
            plt.plot(df['step'], df['smoothed'], label=name, color=colors[i], linewidth=1.7)
            # 阴影可以稍微淡一点，避免干扰主曲线对比
            plt.fill_between(df['step'], 
                             df['smoothed'] - df['std'], 
                             df['smoothed'] + df['std'], 
                             color=colors[i], alpha=0.1)

    if not has_plotted: return

    # plt.title("Training Convergence Comparison", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Training Episodes", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    # 2. 调整刻度数字大小 (labelsize) -- 这是新增的
    plt.tick_params(axis='both', labelsize=22)    # 建议调大到 18-22
    plt.xlim(0, MAX_STEPS)
    #设置刻度间距
    plt.xticks(range(0, MAX_STEPS + 1, 250))
    
    # 应用手动设置的 Y 轴范围
    if Y_LIMITS:
        plt.ylim(Y_LIMITS[0], Y_LIMITS[1])
        print(f"\n⚡ 已锁定 Y 轴范围: {Y_LIMITS}。如果不合适，请在代码开头修改 Y_LIMITS 数值！")
    
    plt.legend(loc='lower right', frameon=True, fontsize=18)
    plt.grid(True, which='major', linestyle='--', alpha=0.8) # 加强网格线
    
    plt.tight_layout()
    plt.savefig("compare2.pdf", dpi=300)
    print("✅ 图片已保存为 compare.pdf")
    plt.show()

if __name__ == "__main__":
    plot_zoomed_curves()