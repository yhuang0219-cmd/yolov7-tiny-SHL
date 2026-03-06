
import matplotlib.pyplot as plt

def read_and_extract_column(file_path, column_index):
    epochs = []
    values = []
    
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) > column_index:
                epochs.append(int(parts[0].split(':')[1]))  # 提取训练次数
                values.append(float(parts[column_index]))  # 提取指定列的值
    return epochs, values

# 替换以下路径为你的实际文件路径
file_path_yolov7_tiny = "yolov7_tiny_result.txt"
file_path_our_yolov7_tiny = "our_yolov7_tiny_result.txt"

# 读取两个文件并提取数据
epochs_yolov7_tiny, map_05_yolov7_tiny = read_and_extract_column(file_path_yolov7_tiny, 11)
epochs_our_yolov7_tiny, map_05_our_yolov7_tiny = read_and_extract_column(file_path_our_yolov7_tiny, 11)

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs_yolov7_tiny, map_05_yolov7_tiny, label='yolov7-tiny', marker='o')
plt.plot(epochs_our_yolov7_tiny, map_05_our_yolov7_tiny, label='our-yolov7-tiny', marker='o')

# 添加图例
plt.legend()

# 添加标题和坐标轴标签
plt.title('map@0.5 ')
plt.xlabel('epoch')
plt.ylabel('map@0.5')

# 显示图表
plt.grid(True)
plt.show()
