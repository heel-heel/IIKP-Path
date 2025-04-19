import pandas as pd
import numpy as np
import os

# 设置基础路径
base_path = "../../../Datasets"

# 读取 clean.csv 文件，假设 V2 列是目标列
clean_file_path = os.path.join(base_path, "M4-Monthly", "clean.csv")
clean_data = pd.read_csv(clean_file_path)

# 提取 V2 列的数据
clean_v2 = clean_data['V2'].values

# 读取 dirty.csv 文件，假设 V2 列是目标列
dirty_file_path = os.path.join(base_path, "M4-Monthly", "Mechanism", "timeseries", "decompose_replace", "trend", "dirty-trend_gain-50.csv")
dirty_data = pd.read_csv(dirty_file_path)

# 提取 V2 列的数据
dirty_v2 = dirty_data['V2'].values

# 确保两个数据集的长度相同
if len(clean_v2) != len(dirty_v2):
    raise ValueError("clean.csv 和 dirty.csv 中的数据点数量不一致")

# 计算每个对应数据点之间的距离
distances = np.abs(dirty_v2 - clean_v2)

# 找出距离最大的前5个索引
top_5_indices = np.argsort(distances)[-5:][::-1]

# 输出结果
for index in top_5_indices:
    distance = distances[index]
    clean_value = clean_v2[index]
    dirty_value = dirty_v2[index]
    print(f"索引: {index}, 距离: {distance}, clean.csv 中的值: {clean_value}, dirty.csv 中的值: {dirty_value}")