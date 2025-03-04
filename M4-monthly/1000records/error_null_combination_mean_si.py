import pandas as pd
import numpy as np
import os
import time

# Mean相关
save_path_mean = './combination/null-mean-si/intermediate/'
if not os.path.exists(save_path_mean):
    os.makedirs(save_path_mean)
read_path_mean = './null/'

# SI相关
save_path_si = './combination/null-mean-si'
if not os.path.exists(save_path_si):
    os.makedirs(save_path_si)
read_path_si = './combination/null-mean-si/intermediate/'

log_file_path = os.path.join(save_path_si, "mean-si-time-log.txt")


def process_and_fill_csv_mean(file_name, output_file_name, fill_rate):
    df = pd.read_csv(os.path.join(read_path_mean, file_name))
    global_mean = df['V2'].mean()
    missing_indices = df[df['V2'].isnull()].index
    np.random.seed(42)
    partial_missing_indices = np.random.choice(missing_indices, size=int(len(missing_indices) * fill_rate), replace=False)
    df.loc[partial_missing_indices, 'V2'] = global_mean
    df.to_csv(os.path.join(save_path_mean, output_file_name), index=False)
    print(f"{output_file_name}已保存到{save_path_mean}")

def process_and_fill_csv_si(file_name, output_file_name):
    df = pd.read_csv(os.path.join(read_path_si, file_name))
    X = df.drop('V1', axis=1).values

    max_iter = 100  # 最大迭代次数
    epsilon = 1e-5  # 平均差异阈值
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]  # 不同的阈值

    X_filled = np.copy(X)
    missing_mask = np.isnan(X_filled)
    X_filled[missing_mask] = 0

    filled_results = []
    for threshold in thresholds:
        X_imputed = np.copy(X_filled)
        for _ in range(max_iter):
            U, s, Vt = np.linalg.svd(X_imputed, full_matrices=False)
            s_thresh = np.maximum(s - threshold, 0)
            X_imputed = U @ np.diag(s_thresh) @ Vt
            avg_diff = np.mean(np.abs(X_imputed[missing_mask] - X_filled[missing_mask]))
            X_filled[missing_mask] = X_imputed[missing_mask]
            if avg_diff < epsilon:
                break
        filled_results.append(X_imputed)
    best_filled = min(filled_results, key=lambda x: np.mean(np.abs(x[~missing_mask] - X[~missing_mask])))
    df.loc[missing_mask[:, 0], 'V2'] = best_filled[missing_mask[:, 0], 1]
    df.to_csv(os.path.join(save_path_si, output_file_name), index=False)
    print(f"{output_file_name}已保存到{save_path_si}")

Missing_rate = [10, 30, 50, 70, 90]
Fill_rate = [0.1, 0.3, 0.5, 0.7, 0.9]
with open(log_file_path, "w") as log_file:
    for rate in Missing_rate:
        input_file_mean = f'dirty-{rate}.csv'
        for fill_rate in Fill_rate:
            intermediate_file = f'intermediate-mean-{rate}-fill{int(fill_rate * 100)}.csv'
            start_time = time.time()
            process_and_fill_csv_mean(input_file_mean, intermediate_file, fill_rate)
            end_time = time.time()
            processing_time = end_time - start_time
            log_file.write(f"Mean processing {input_file_mean} with fill rate {fill_rate}: {processing_time:.4f} seconds.\n")

            output_file = f'dirty-mean-si-{rate}-fill{int(fill_rate * 100)}.csv'
            start_time = time.time()
            process_and_fill_csv_si(intermediate_file, output_file)
            end_time = time.time()
            processing_time = end_time - start_time
            log_file.write(f"SI processing {intermediate_file}: {processing_time:.4f} seconds.\n")