import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

save_path='./null-city-si/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def process_and_fill_csv(file_name, output_file_name):
    df_copy = pd.read_csv(file_name)
    df = df_copy.replace("", pd.NA)
    missing_indices = df[df['city'].isnull()].index

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    df_drop = df.drop(missing_indices)
    known_city_codes = df_drop['city'].values
    # 初始化缺失值为0
    X = df.drop(columns=['id']).values
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

            # 计算平均差异
            avg_diff = np.mean(np.abs(X_imputed[missing_indices] - X_filled[missing_indices]))

            # 更新V2列的缺失值
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
        min_code = np.min(known_city_codes)
        max_code = np.max(known_city_codes)

        # 如果生成的数值超出范围，选择最接近的有效编码
        if generated_value < min_code:
            generated_value = int(min_code)
        elif generated_value > max_code:
            generated_value = int(max_code)
        else:
            distances = np.abs(known_city_codes - generated_value)
            closest_index = np.argmin(distances)
            generated_value = int(known_city_codes[closest_index])
        city_name = label_encoders['city'].inverse_transform([int(generated_value)])[0]
        df_copy.at[index, 'city'] = city_name
        #print(city_name)
    df_copy.to_csv(os.path.join(save_path,output_file_name), index=False)

Missing_rate = [10, 30, 50, 70, 90]
for rate in Missing_rate:
    input_file = f'dirty-city-{rate}.csv'
    output_file = f'dirty-city-si-{rate}.csv'
    process_and_fill_csv(input_file, output_file)