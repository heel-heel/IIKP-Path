import pandas as pd
import numpy as np
import random

# 读取数据
df = pd.read_csv('clean.csv')


# 定义一个函数来生成具有特定百分比缺失值的 DataFrame 并保存为 CSV
def generate_dirty_csv(percent_missing):
    # 计算需要设置为缺失的行数
    num_missing = int(len(df) * percent_missing / 100)

    # 随机选择索引
    random_indices = random.sample(range(len(df)), num_missing)

    # 设置随机选择的索引处的 'City' 值为 ""
    #df.loc[random_indices, 'City'] = ""#分类
    df.loc[random_indices, 'Score'] = ""#回归

    # 保存到 CSV 文件
    filename = f'dirty-{percent_missing}.csv'
    df.to_csv(filename, index=False)
    print(f'File {filename} generated with {percent_missing}% missing values.')


# 生成不同百分比缺失值的 CSV 文件
percent_missing_values = [10, 30, 50, 70, 90]
for percent in percent_missing_values:
    generate_dirty_csv(percent)