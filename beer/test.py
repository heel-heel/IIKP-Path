import pandas as pd
import numpy as np
from scipy.optimize import minimize

# 导入CSV文件
file_path = 'dirty-10.csv'  # 输入文件路径
data = pd.read_csv(file_path)

# 检查数据
print("原始数据：")
print(data.head())

# 提取数值属性列（假设V1是非数值属性，其他列是数值属性）
numeric_data = data.drop(columns=['V1'])  # 去掉非数值属性列
V2_index = numeric_data.columns.get_loc('V2')  # 获取V2列的索引
X = numeric_data.values  # 转换为numpy数组

# 获取V2列的缺失值位置
missing_mask = np.isnan(X[:, V2_index])
M = np.ones_like(X[:, V2_index])  # 创建掩码矩阵
M[missing_mask] = 0

# 初始化参数
p = 5  # 潜因子维度
n, d = X.shape
U = np.random.rand(n, p)  # 初始化U矩阵
V = np.random.rand(d, p)  # 初始化V矩阵
cmax = 100  # 最大迭代次数
threshold = 1e-4  # 收敛阈值

# 定义优化目标函数
def objective(params):
    # 将一维数组重新构造为U和V矩阵
    U = params[:n * p].reshape(n, p)
    V = params[n * p:].reshape(d, p)
    X_pred = U @ V.T
    observed_diff = np.sum((X[M == 1] - X_pred[M == 1]) ** 2)
    return observed_diff

# 定义优化过程
def optimize(U, V):
    # 将U和V展平为一维数组
    params = np.concatenate([U.flatten(), V.flatten()])
    result = minimize(objective, params, method='BFGS', options={'maxiter': cmax})
    # 重新构造U和V矩阵
    U_opt = result.x[:n * p].reshape(n, p)
    V_opt = result.x[n * p:].reshape(d, p)
    return U_opt, V_opt

# 迭代优化
consecutive_count = 0  # 用于记录连续满足条件的次数
previous_diffs = []  # 用于存储最近5次的avg_diff值

for i in range(cmax):
    print("--------------------")
    print(f"echo:{i}")
    U, V = optimize(U, V)
    X_pred = U @ V.T
    avg_diff = np.mean((X[M == 1] - X_pred[M == 1]) ** 2)
    print(avg_diff)

    # 更新previous_diffs列表，保持其长度为5
    previous_diffs.append(avg_diff)
    if len(previous_diffs) > 5:
        previous_diffs.pop(0)  # 移除最早的值，保持列表长度为5

    # 检查连续5次的差值是否都小于1
    if len(previous_diffs) == 5:
        diffs = [abs(previous_diffs[j] - previous_diffs[j + 1]) for j in range(len(previous_diffs) - 1)]
        if all(diff < 1 for diff in diffs):
            consecutive_count += 1
            if consecutive_count >= 5:
                print("连续5次循环的差值都小于1，停止优化。")
                break
        else:
            consecutive_count = 0  # 如果不满足条件，重置计数器

# 填充缺失值
X_imputed = X.copy()
X_imputed[missing_mask] = (U @ V.T)[missing_mask]

# 将填充后的V2列数据更新到原始数据中
data.iloc[:, V2_index + 1] = X_imputed[:, V2_index]  # 加1是因为V1列被去掉了

# 导出填充后的CSV文件
output_path = 'output.csv'  # 输出文件路径
data.to_csv(output_path, index=False)

print("填充后的数据：")
print(data.head())
print(f"填充后的数据已导出到 {output_path}")