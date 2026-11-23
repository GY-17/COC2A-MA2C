import tensorflow as tf
from tensorflow.core.util import event_pb2
import os

# 配置文件名
# 请确保这里的文件名与您本地的原始文件名完全一致
input_file = r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\compare\ma2c30nodes_Entropy_annealing\experiment_20251119-112851\episode\events.out.tfevents.1763522932.autodl-container-c10245bd8d-3b243c4f.3967.1.v2"
output_file = r"E:\python_object\VRPD\2pre_reward_Entropy_annealing_maca2c_rolloutlen\logs\compare\ma2c30nodes_Entropy_annealing\experiment_20251119-112851\episode\events.out.tfevents.modified.1763522932.autodl-container-c10245bd8d-3b243c4f.3967.1.v2"
def modify_tag(input_path, output_path):
    print(f"正在读取文件: {input_path} ...")
    
    try:
        # 创建写入器
        writer = tf.io.TFRecordWriter(output_path)
        # 读取原始数据
        dataset = tf.data.TFRecordDataset(input_path)
        
        count = 0
        modified_count = 0
        
        for raw_record in dataset:
            # 解析事件
            event = event_pb2.Event.FromString(raw_record.numpy())
            
            # 检查并修改标签
            if event.HasField('summary'):
                for value in event.summary.value:
                    if value.tag == 'Episode/Total_Reward':
                        value.tag = 'Episode/Real_Reward'
                        modified_count += 1
            
            # 写入新文件
            writer.write(event.SerializeToString())
            count += 1
            
        writer.close()
        print(f"处理完成！")
        print(f"- 共扫描事件数: {count}")
        print(f"- 已修改标签数: {modified_count}")
        print(f"- 新文件已保存为: {output_path}")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    if os.path.exists(input_file):
        modify_tag(input_file, output_file)
    else:
        print(f"错误: 在当前目录下未找到文件 '{input_file}'")
        print("请检查文件名是否正确，或将文件路径修改为绝对路径。")