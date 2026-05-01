# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize


def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    data = pd.read_csv(input_file)
    if nonnumerical_column != "None":
        numeric_data = data.drop(columns=[nonnumerical_column])
    else:
        numeric_data = data

    target_index = numeric_data.columns.get_loc(target_column)
    X = numeric_data.values.astype(np.float64)

    # 创建缺失掩码矩阵（整个矩阵）
    missing_mask_matrix = np.isnan(X)  # 整个矩阵的缺失掩码
    M = np.ones_like(X)  # 观测掩码矩阵
    M[missing_mask_matrix] = 0

    # 只关注目标列的缺失
    target_missing_mask = np.isnan(X[:, target_index])

    # 初始化
    p = 5  # 潜在因子数量
    n, d = X.shape
    U = np.random.rand(n, p)
    V = np.random.rand(d, p)
    cmax = 15
    threshold = 0.01  # 收敛阈值

    # 使用交替最小二乘法（ALS）
    def optimize_als(U, V, max_iter=10):
        """使用交替最小二乘法优化"""
        U_opt = U.copy()
        V_opt = V.copy()

        for iteration in range(max_iter):
            # 更新U
            for i in range(n):
                # 找到有观测值的特征
                observed_j = np.where(M[i, :] == 1)[0]
                if len(observed_j) > 0:
                    V_observed = V_opt[observed_j, :]
                    X_observed = X[i, observed_j]

                    # 最小二乘解
                    A = V_observed.T @ V_observed
                    b = V_observed.T @ X_observed

                    # 添加正则化
                    A += 1e-6 * np.eye(p)
                    try:
                        U_opt[i] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        # 如果求解失败，使用伪逆
                        U_opt[i] = np.linalg.pinv(A) @ b

            # 更新V
            for j in range(d):
                observed_i = np.where(M[:, j] == 1)[0]
                if len(observed_i) > 0:
                    U_observed = U_opt[observed_i, :]
                    X_observed = X[observed_i, j]

                    # 最小二乘解
                    A = U_observed.T @ U_observed
                    b = U_observed.T @ X_observed

                    # 添加正则化
                    A += 1e-6 * np.eye(p)
                    try:
                        V_opt[j] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        # 如果求解失败，使用伪逆
                        V_opt[j] = np.linalg.pinv(A) @ b

        return U_opt, V_opt

    previous_diffs = []

    for i in range(cmax):
        print("--------------------")
        print(f"echo:{i}")

        # 使用ALS进行优化
        U, V = optimize_als(U, V, max_iter=5)  # 每次迭代5次ALS更新
        X_pred = U @ V.T

        # 只计算观测值的MSE
        observed_mask = M == 1
        avg_diff = np.mean((X[observed_mask] - X_pred[observed_mask]) ** 2)
        print(f"MSE: {avg_diff:.6f}")

        previous_diffs.append(avg_diff)
        print(f"Previous diffs: {previous_diffs}")

        if len(previous_diffs) > 5:
            previous_diffs.pop(0)

        # 检查收敛条件
        if len(previous_diffs) == 5:
            diffs = [abs(previous_diffs[j] - previous_diffs[j + 1])
                     for j in range(len(previous_diffs) - 1)]
            print(f"Consecutive diffs: {diffs}")
            if all(diff < threshold for diff in diffs):
                print("Stop iteration when the difference has been less than threshold for 5 consecutive times")
                break

    # 只插补目标列的缺失值
    X_imputed = X.copy()
    X_imputed[target_missing_mask, target_index] = (U @ V.T)[target_missing_mask, target_index]

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