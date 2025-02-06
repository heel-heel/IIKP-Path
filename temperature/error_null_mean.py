import pandas as pd
import numpy as np

# 读取CSV文件
data = pd.read_csv('clean.csv')

# 随机选择10%的数据点设置为NaN
indices_to_nan = np.random.choice(data.index, size=int(len(data) * 0.9), replace=False)
data.loc[indices_to_nan, 'Temp'] = np.nan

# 计算Ozone列的平均值（除去NaN）
average_ozone = data['Temp'].mean()

# 定义一个函数来填充NaN值
def fill_nan_with_neighbors(index, data, average_ozone):
    row = data.iloc[index]
    if pd.isna(row['Temp']):
        # 寻找前面的最近的非空值
        previous = np.nan
        for i in range(index-1, -1, -1):
            if not pd.isna(data['Temp'][i]):
                previous = data['Temp'][i]
                break
        # 寻找后面的最近的非空值
        next = np.nan
        for i in range(index+1, len(data)):
            if not pd.isna(data['Temp'][i]):
                next = data['Temp'][i]
                break
#        if pd.isna(previous) and pd.isna(next):
#            return average_ozone
        if pd.isna(previous):
            return next
        elif pd.isna(next):
            return previous
        else:
            return (previous + next) / 2
    return row['Temp']

# 应用函数填充NaN值
data['Temp'] = data.index.map(lambda x: fill_nan_with_neighbors(x, data, average_ozone))

# 保存处理后的数据
data.to_csv('dirty-mean-90.csv', index=False)