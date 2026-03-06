import matplotlib.pyplot as plt


# 定义一个函数来读取文件并提取每行的第11个字符
def extract_map_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 提取每行的第11个字符，并转换为浮点数
    map_values = []
    for line in lines:
        characters = line.split()
        if len(characters) >= 11:  # 确保有至少11个字符
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
plt.plot(epochs, map1, label='Dataset 1 (mAP@0.5)', marker='o')
plt.plot(epochs, map2, label='Dataset 2 (mAP@0.5)', marker='x')

# 添加标签和标题
plt.xlabel('Epoch')
plt.ylabel('mAP@0.5')
plt.title('mAP@0.5 vs Epoch')

# 移除网格
plt.grid(False)

# 将图例移动到右下角
plt.legend(loc='lower right')

# 保存图片
plt.savefig('map_comparison.png')

# 显示图表
plt.show()
