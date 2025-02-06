import pandas as pd
import numpy as np

# 读取CSV文件
data = pd.read_csv('clean.csv')

global_median = data['Sunspots'].median()
# 随机选择10%的数据点设置为NaN
indices_to_nan = np.random.choice(data.index, size=int(len(data) * 0.9), replace=False)
data.loc[indices_to_nan, 'Sunspots'] = np.nan

# 使用简单移动平均（SMA）方法填充缺失值，采用中位数
# 这里设置窗口大小为5，可以根据需要调整
window_size = 5
data['Sunspots'] = data['Sunspots'].fillna(data['Sunspots'].rolling(window=window_size, min_periods=1).median())

data['Sunspots'] = data['Sunspots'].fillna(global_median)
# 保存处理后的数据
data.to_csv('dirty-median-90.csv', index=False)