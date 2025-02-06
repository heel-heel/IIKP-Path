import pandas as pd
import numpy as np

# 读取CSV文件
data = pd.read_csv('clean.csv')

# 计算需要设置为空值的记录数量
num_records_to_nan = int(len(data) * 0.9)

# 随机选择一个起始索引
random_start_index = np.random.randint(0, len(data) - num_records_to_nan + 1)

# 选择连续的10%记录的索引
start_index = random_start_index
end_index = start_index + num_records_to_nan

# 将选中的记录的Ozone值设置为NaN
data.loc[start_index:end_index, 'Ozone'] = pd.NA

# 导出文件
data.to_csv('dirty-continuous-90.csv', index=False)