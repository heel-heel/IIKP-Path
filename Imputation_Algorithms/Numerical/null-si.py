import pandas as pd
import numpy as np
import os
import time


def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    df = pd.read_csv(input_file)
    X = df.drop(nonnumerical_column, axis=1).values

    max_iter = 100
    epsilon = 1e-5
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]

    X_filled = np.copy(X)
    missing_mask = np.isnan(X_filled)
    missing_mask_target = missing_mask[:, 0]
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
            X_filled[missing_mask_target, 1] = X_imputed[missing_mask_target, 1]

            if avg_diff < epsilon:
                break

        filled_results.append(X_imputed)

    best_filled = min(filled_results, key=lambda x: np.mean(np.abs(x[~missing_mask] - X[~missing_mask])))
    df.loc[missing_mask_target, 'V2'] = best_filled[missing_mask_target, 1]
    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')



if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)