import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

save_path='./null-mfi/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

read_path = './null/'

def process_and_fill_csv(file_name, output_file_name):
    data = pd.read_csv(os.path.join(read_path,file_name))
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
    threshold = 1  # 收敛阈值

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
        print(previous_diffs)
        if len(previous_diffs) > 5:
            previous_diffs.pop(0)  # 移除最早的值，保持列表长度为5

        # 检查连续5次的差值是否都小于1
        if len(previous_diffs) == 5:
            diffs = [abs(previous_diffs[j] - previous_diffs[j + 1]) for j in range(len(previous_diffs) - 1)]
            print(diffs)
            if all(diff < threshold for diff in diffs):
                print("连续5次的差值都小于1，停止迭代")
                break

    # 填充缺失值
    X_imputed = X.copy()
    X_imputed[missing_mask] = (U @ V.T)[missing_mask]
    data.iloc[:, V2_index + 1] = X_imputed[:, V2_index]
    data.to_csv(os.path.join(save_path,output_file_name), index=False)
    print(f"填充后的数据已导出到 {output_file_name}")

Missing_rate = [10, 30, 50, 70, 90]
for rate in Missing_rate:
    input_file = f'dirty-{rate}.csv'
    output_file = f'dirty-mfi-{rate}.csv'
    process_and_fill_csv(input_file, output_file)