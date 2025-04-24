import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data1 = pd.read_csv(os.path.join("../../../datasets", "M4-Monthly", "clean.csv"))
data2 = pd.read_csv(os.path.join("../../../datasets", "M4-Monthly", "Mechanism", "timeseries", "decompose_replace", "trend", "dirty-trend_gain-50.csv"))

# 确保数据集中的V2列存在
if 'V2' not in data1.columns or 'V2' not in data2.columns:
    raise ValueError("数据集中不存在 'V2' 列")

# 提取V2列的数据
v2_data1 = data1['V2'].values
v2_data2 = data2['V2'].values

# 确保两个数据集的长度相同
if len(v2_data1) != len(v2_data2):
    raise ValueError("两个数据集的长度不一致")

# 计算MSE（均方误差）
mse = mean_squared_error(v2_data1, v2_data2)
print(f"MSE (Mean Squared Error): {mse}")

# 计算MAE（平均绝对误差）
mae = mean_absolute_error(v2_data1, v2_data2)
print(f"MAE (Mean Absolute Error): {mae}")

# 计算RMSE（均方根误差）
rmse = np.sqrt(mse)
print(f"RMSE (Root Mean Squared Error): {rmse}")

# 计算相关系数
correlation_coefficient = np.corrcoef(v2_data1, v2_data2)[0, 1]
print(f"Correlation Coefficient: {correlation_coefficient}")