import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('clean.csv')


# 定义一个函数来设置随机离群值
def set_random_outliers(df, column, outlier_percentage, outlier_scale_factor):
    # 获取当前Temp列的最大值和最小值
    temp_max = df[column].max()
    temp_min = df[column].min()

    # 计算需要设置为离群值的数据点数量
    num_outliers = int(len(df) * outlier_percentage / 100)

    # 随机选择数据点
    indices = np.random.choice(df.index, num_outliers, replace=False)

    # 初始化离群值列表
    outlier_values = []

    # 对于每个选定的离群点，随机决定是偏高还是偏低，并生成离群值
    for _ in range(num_outliers):
        # 随机决定是生成高于最大值的离群值还是低于最小值的离群值
        # 可以调整prob的值来控制偏高和偏低的比例
        prob = 0.5  # 50%的概率偏高，50%的概率偏低
        if np.random.rand() < prob:
            # 生成偏高的离群值
            outlier_value = temp_max + np.random.rand() * outlier_scale_factor * (temp_max - temp_min)
        else:
            # 生成偏低的离群值
            outlier_value = temp_min - np.random.rand() * outlier_scale_factor * (temp_max - temp_min)

        outlier_values.append(outlier_value)

        # 将这些点的Temp值设置为随机离群值
    df.loc[indices, column] = outlier_values

    return df


# 设置离群值
outlier_percentages = [10, 30, 50, 70, 90]  # 不同的离群值比例
outlier_scale_factors = [20]  # 控制离群值偏离正常范围的程度

for outlier_percentage in outlier_percentages:
    for outlier_scale_factor in outlier_scale_factors:
        # 复制原始数据帧以避免修改原始数据
        modified_df = df.copy()

        # 设置随机离群值
        modified_df = set_random_outliers(modified_df, 'Temp', outlier_percentage, outlier_scale_factor)

        # 保存修改后的数据帧到新的CSV文件
        filename = f'error_{outlier_percentage}.csv'
        modified_df.to_csv(filename, index=False)