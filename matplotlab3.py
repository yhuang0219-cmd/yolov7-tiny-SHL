import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置字体为支持中文的字体，例如 "SimHei"
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义一个函数来读取文件并提取每行的第11个字符
def extract_map_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 提取每行的第11个字符，并转换为浮点数
    map_values = []
    for line in lines:
        characters = line.split()
        if len(characters) >= 12:  # 确保有至少11个字符
            try:
                map_values.append(float(characters[10]))  # 第11个字符索引为10
            except ValueError:
                continue  # 跳过无法转换为浮点数的值
    return map_values

# 从两个文件中提取 mAP@0.5 数据
map1 = extract_map_from_file('runs/test/yuanmx.txt')
map2 = extract_map_from_file('runs/test/xiugaihou.txt')

# 生成训练次数序列（假设每个文件的行数相同）
epochs = list(range(1, len(map1) + 1))

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(epochs, map1, label='原yolov7-tiny', marker='o')
plt.plot(epochs, map2, label='修改后yolov7-tiny', marker='x')

# 添加标签和标题
plt.xlabel('训练次数', fontsize=20)  # 增加字体大小
plt.ylabel('mAP@0.5', fontsize=24)  # 增加字体大小
# plt.title('mAP@0.5:0.95对比', fontsize=16)  # 添加标题并设置字体大小

# 移除网格
plt.grid(False)

# 设置图例字体大小
plt.legend(loc='lower right', fontsize=20)

# 设置刻度标签字体大小
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# 保存图片
plt.savefig('map_comparison11.png')

# 显示图表
plt.show()
