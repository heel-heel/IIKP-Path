import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def process_and_fill(input_file, output_file, target_column, unrelated_column):
    df_copy = pd.read_csv(input_file)
    if 'quality' in df_copy.columns:
        df_copy = pd.read_csv(input_file, dtype={'quality': 'object'})
    elif 'Type' in df_copy.columns:
        df_copy = pd.read_csv(input_file, dtype={'Type': 'object'})
    else:
        df_copy = pd.read_csv(input_file)
    df = df_copy.copy()
    missing_indices = df[df[target_column].isnull()].index

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    df_drop = df.drop(missing_indices)
    known_target_codes = df_drop[target_column].values
    # 初始化缺失值为0
    if unrelated_column != "None":
        X = df.drop(columns=[unrelated_column]).values
    else:
        X = df.values
    X_filled = np.copy(X)
    missing_mask = np.isnan(X_filled)
    missing_mask_V2 = missing_mask[:, 0]
    X_filled[missing_mask] = 0

    # 初始化参数
    max_iter = 100  # 最大迭代次数
    epsilon = 1e-5  # 平均差异阈值
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]  # 不同的阈值

    # 存储不同阈值下的填补结果
    filled_results = []

    for threshold in thresholds:
        X_imputed = np.copy(X_filled)
        for _ in range(max_iter):
            # 计算软阈值奇异值分解
            U, s, Vt = np.linalg.svd(X_imputed, full_matrices=False)
            s_thresh = np.maximum(s - threshold, 0)
            X_imputed = U @ np.diag(s_thresh) @ Vt
            avg_diff = np.mean(np.abs(X_imputed[missing_indices] - X_filled[missing_indices]))
            X_filled[missing_indices, 0] = X_imputed[missing_indices, 0]

            # 判断是否停止迭代
            if avg_diff < epsilon:
                break

        filled_results.append(X_imputed)  # 保存当前阈值下的最终结果

    # 选择最优填补结果
    best_filled = min(filled_results, key=lambda x: np.mean(np.abs(x[~missing_indices] - X[~missing_indices])))

    for index in missing_indices:
        generated_value = best_filled[index, 0]
        #print("------------")
        #print(generated_value)

        # 确保生成的数值在已知的城市编码范围内
        min_code = np.min(known_target_codes)
        max_code = np.max(known_target_codes)

        # 如果生成的数值超出范围，选择最接近的有效编码
        if generated_value < min_code:
            generated_value = int(min_code)
        elif generated_value > max_code:
            generated_value = int(max_code)
        else:
            distances = np.abs(known_target_codes - generated_value)
            closest_index = np.argmin(distances)
            generated_value = int(known_target_codes[closest_index])
        target_name = label_encoders[target_column].inverse_transform([int(generated_value)])[0]
        df_copy.at[index, target_column] = target_name
    df_copy.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')



if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    unrelated_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, unrelated_column)