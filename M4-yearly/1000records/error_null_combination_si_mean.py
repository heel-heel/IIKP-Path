import pandas as pd
import numpy as np
import os
import time

# SI相关
save_path_si = './combination/null-si-mean/intermediate'
if not os.path.exists(save_path_si):
    os.makedirs(save_path_si)

read_path_si = './null/'

# Mean相关
save_path_mean = './combination/null-si-mean'
if not os.path.exists(save_path_mean):
    os.makedirs(save_path_mean)

read_path_mean = './combination/null-si-mean/intermediate'

log_file_path = os.path.join(save_path_mean,"si-mean-time-log.txt")


def process_and_fill_csv_si(file_name, output_file_name, fill_rate):
    df = pd.read_csv(os.path.join(read_path_si, file_name))
    X = df.drop('V1', axis=1).values

    # 初始化参数
    max_iter = 100  # 最大迭代次数
    epsilon = 1e-5  # 平均差异阈值
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]  # 不同的阈值

    # 初始化缺失值为0
    X_filled = np.copy(X)
    missing_mask = np.isnan(X_filled)
    missing_mask_V2 = missing_mask[:, 0]
    X_filled[missing_mask] = 0

    # 随机选择缺失部分的指定比例进行填补
    np.random.seed(42)
    partial_missing_mask = np.random.rand(*missing_mask_V2.shape) < fill_rate
    partial_missing_mask = np.logical_and(missing_mask_V2, partial_missing_mask)

    filled_results = []
    for threshold in thresholds:
        X_imputed = np.copy(X_filled)
        for _ in range(max_iter):
            # 计算软阈值奇异值分解
            U, s, Vt = np.linalg.svd(X_imputed, full_matrices=False)
            s_thresh = np.maximum(s - threshold, 0)
            X_imputed = U @ np.diag(s_thresh) @ Vt

            avg_diff = np.mean(np.abs(X_imputed[partial_missing_mask] - X_filled[partial_missing_mask]))
            X_filled[partial_missing_mask, 1] = X_imputed[partial_missing_mask, 1]
            if avg_diff < epsilon:
                break

        filled_results.append(X_imputed)

    best_filled = min(filled_results, key=lambda x: np.mean(np.abs(x[~missing_mask] - X[~missing_mask])))
    df.loc[partial_missing_mask, 'V2'] = best_filled[partial_missing_mask, 1]
    print(f"{output_file_name}已保存到{save_path_si}")
    df.to_csv(os.path.join(save_path_si, output_file_name), index=False)


def process_and_fill_csv_mean(file_name, output_file_name):
    data = pd.read_csv(os.path.join(read_path_mean, file_name))
    global_mean = data['V2'].mean()
    data['V2'] = data['V2'].fillna(global_mean)
    data.to_csv(os.path.join(save_path_mean, output_file_name), index=False)
    print(f"{output_file_name}已保存到{save_path_mean}")


Missing_rate = [10, 30, 50, 70, 90]
Fill_rate = [0.1, 0.3, 0.5, 0.7, 0.9]
with open(log_file_path, "w") as log_file:
    # SI填补：对每个缺失率和填补比例生成中间文件
    for rate in Missing_rate:
        input_file_si = f'dirty-{rate}.csv'
        for fill_rate in Fill_rate:
            output_file_si = f'intermediate-si-{rate}-fill{int(fill_rate * 100)}.csv'
            start_time = time.time()
            process_and_fill_csv_si(input_file_si, output_file_si, fill_rate)
            end_time = time.time()
            processing_time = end_time - start_time
            log_file.write(f"{input_file_si}-{fill_rate}:{processing_time:.4f} seconds.\n")

    # 均值填补：对所有中间文件进行均值填补
    for rate in Missing_rate:
        for fill_rate in Fill_rate:
            input_file_mean = f'intermediate-si-{rate}-fill{int(fill_rate * 100)}.csv'
            output_file_mean = f'dirty-si-mean-{rate}-fill{int(fill_rate * 100)}.csv'
            start_time = time.time()
            process_and_fill_csv_mean(input_file_mean, output_file_mean)
            end_time = time.time()
            processing_time = end_time - start_time
            log_file.write(f"{input_file_mean}-{fill_rate}:{processing_time:.4f} seconds.\n")