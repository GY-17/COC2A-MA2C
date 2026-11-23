import tensorflow as tf
from tensorflow.core.util import event_pb2
import os

# ================= 配置区域 =================
# 输入文件路径 (请使用 r"" 原始字符串格式防止路径乱码)
input_file = r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\compare\MACA2C\episode\events.out.tfevents.1764593492.autodl-container-d7fa43adb1-4548ccbc.12380.1.v2"

# 输出文件路径
output_file = r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\compare\MACA2C\episode\events.out.tfevents.filtered_2000.v2"

# 截断的步数阈值
CUTOFF_STEP = 2000
# ===========================================

def filter_events(input_path, output_path, cutoff):
    print(f"正在读取文件: {input_path} ...")
    
    try:
        # 创建写入器
        writer = tf.io.TFRecordWriter(output_path)
        # 读取原始数据
        dataset = tf.data.TFRecordDataset(input_path)
        
        total_count = 0
        kept_count = 0
        
        for raw_record in dataset:
            total_count += 1
            # 解析事件
            event = event_pb2.Event.FromString(raw_record.numpy())
            
            # 核心逻辑：只保留 step 小于等于 cutoff 的事件
            if event.step <= cutoff:
                writer.write(event.SerializeToString())
                kept_count += 1
            
        writer.close()
        print("-" * 30)
        print(f"处理完成！")
        print(f"- 原始事件总数: {total_count}")
        print(f"- 保留事件数 (Step <= {cutoff}): {kept_count}")
        print(f"- 新文件已保存为: {output_path}")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 检查输入文件是否存在
    if os.path.exists(input_file):
        filter_events(input_file, output_file, CUTOFF_STEP)
    else:
        print(f"错误: 找不到文件")
        print(f"路径: {input_file}")
        print("提示: 请检查路径是否正确，并确保使用了 r'...' 格式。")