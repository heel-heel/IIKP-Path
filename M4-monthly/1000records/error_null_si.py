import pandas as pd
import numpy as np
import os
import time

save_path = './null-si/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

read_path = './null/'

def process_and_fill_csv(file_name, output_file_name):
    df = pd.read_csv(os.path.join(read_path, file_name))
    X = df.drop('V1', axis=1).values

    max_iter = 100
    epsilon = 1e-5
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]

    X_filled = np.copy(X)
    missing_mask = np.isnan(X_filled)
    missing_mask_V2 = missing_mask[:, 0]
    X_filled[missing_mask] = 0

    filled_results = []
    for threshold in thresholds:
        X_imputed = np.copy(X_filled)
        for _ in range(max_iter):
            # 计算软阈值奇异值分解
            U, s, Vt = np.linalg.svd(X_imputed, full_matrices=False)
            s_thresh = np.maximum(s - threshold, 0)
            X_imputed = U @ np.diag(s_thresh) @ Vt

            avg_diff = np.mean(np.abs(X_imputed[missing_mask] - X_filled[missing_mask]))
            X_filled[missing_mask_V2, 1] = X_imputed[missing_mask_V2, 1]

            if avg_diff < epsilon:
                break

        filled_results.append(X_imputed)

    best_filled = min(filled_results, key=lambda x: np.mean(np.abs(x[~missing_mask] - X[~missing_mask])))
    df.loc[missing_mask_V2, 'V2'] = best_filled[missing_mask_V2, 1]
    print(f"{output_file_name}已保存到{save_path}")
    df.to_csv(os.path.join(save_path, output_file_name), index=False)

log_file_path = os.path.join(save_path, "si-time-log.txt")
with open(log_file_path, "w") as log_file:
    Missing_rate = [10, 30, 50, 70, 90]
    for rate in Missing_rate:
        input_file = f'dirty-{rate}.csv'
        output_file = f'dirty-si-{rate}.csv'
        start_time = time.time()
        process_and_fill_csv(input_file, output_file)
        end_time = time.time()
        processing_time = end_time - start_time
        log_file.write(f"{input_file}:{processing_time:.4f} seconds.\n")