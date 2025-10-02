import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    data = pd.read_csv(input_file)
    if nonnumerical_column != "None":
        numeric_data = data.drop(columns=[nonnumerical_column])  # 去掉非数值属性列
    else:
        numeric_data = data
    target_index = numeric_data.columns.get_loc(target_column)  # 获取V2列的索引
    X = numeric_data.values  # 转换为numpy数组

    # 获取V2列的缺失值位置
    missing_mask = np.isnan(X[:, target_index])
    M = np.ones_like(X[:, target_index])  # 创建掩码矩阵
    M[missing_mask] = 0

    # 初始化参数
    p = 5  # 潜因子维度
    n, d = X.shape
    U = np.random.rand(n, p)  # 初始化U矩阵
    V = np.random.rand(d, p)  # 初始化V矩阵
    cmax = 100  # 最大迭代次数
    #threshold = 10000000  # 收敛阈值,可调整
    #threshold = 0.0001
    threshold = 1

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
    if nonnumerical_column != "None":
        data.iloc[:, target_index + 1] = X_imputed[:, target_index]
    else:
        data.iloc[:, target_index] = X_imputed[:, target_index]
    data.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')



if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)