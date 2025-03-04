import pandas as pd
import os
import numpy as np
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# IIM相关
save_path_iim = './combination/null-iim-mean/intermediate'
if not os.path.exists(save_path_iim):
    os.makedirs(save_path_iim)
read_path = './null/'

# Mean相关
save_path_mean = './combination/null-iim-mean'
if not os.path.exists(save_path_mean):
    os.makedirs(save_path_mean)
read_path_mean = './combination/null-iim-mean/intermediate'

log_file_path = os.path.join(save_path_mean, "iim-mean-time-log.txt")

def process_and_fill_csv_iim(file_name, output_file_name, fill_rate):
    df_copy = pd.read_csv(os.path.join(read_path, file_name))
    df = df_copy.drop(columns='V1')
    v2_observed_count = df['V2'].notnull().sum()
    missing_indices = df[df['V2'].isnull()].index

    np.random.seed(42)
    partial_missing_indices = np.random.choice(missing_indices, size=int(len(missing_indices) * fill_rate), replace=False)
    partial_missing_mask = df.index.isin(partial_missing_indices)

    df_filled = df.copy()
    def impute_iim(df_filled):
        df_imputed = df_filled.copy()
        feature = 'V2'
        for i in partial_missing_indices:
            if pd.isnull(df.iloc[i][feature]):
                models = []
                errors = []
                for k in range(5, v2_observed_count + 1, 10):
                    knn = KNeighborsRegressor(n_neighbors=k)
                    X_train = df_filled.dropna(subset=[feature]).drop(columns=[feature])
                    y_train = df_filled.dropna(subset=[feature])[feature]
                    knn.fit(X_train.values, y_train.values)
                    distances, indices = knn.kneighbors(df_filled.iloc[i].drop(feature).values.reshape(1, -1))

                    X_train_neighbor = X_train.iloc[indices[0]]
                    y_train_neighbor = y_train.iloc[indices[0]]
                    model = LinearRegression()
                    model.fit(X_train_neighbor, y_train_neighbor)
                    models.append(model)

                    y_pred_neighbor = model.predict(X_train_neighbor)
                    error = mean_squared_error(y_train_neighbor, y_pred_neighbor)
                    errors.append(error)

                best_model_idx = errors.index(min(errors))
                best_model = models[best_model_idx]
                df_imputed.at[i, feature] = best_model.predict(df_filled.iloc[i].drop(feature).to_frame().T)[0]

        return df_imputed
    df_imputed = impute_iim(df_filled)

    for index in partial_missing_indices:
        generated_value = df_imputed.at[index, 'V2']
        df_copy.at[index, 'V2'] = generated_value

    df_copy.to_csv(os.path.join(save_path_iim, output_file_name), index=False)
    print(f"{output_file_name}已保存到{save_path_iim}")

def process_and_fill_csv_mean(file_name, output_file_name):
    data = pd.read_csv(os.path.join(read_path_mean, file_name))
    global_mean = data['V2'].mean()
    data['V2'] = data['V2'].fillna(global_mean)
    data.to_csv(os.path.join(save_path_mean, output_file_name), index=False)
    print(f"{output_file_name}已保存到{save_path_mean}")

Missing_rate = [10, 30, 50, 70, 90]
Fill_rate = [0.1, 0.3, 0.5, 0.7, 0.9]
with open(log_file_path, "w") as log_file:
    # IIM填补：对每个缺失率和填补比例生成中间文件
    for rate in Missing_rate:
        input_file_iim = f'dirty-{rate}.csv'
        for fill_rate in Fill_rate:
            output_file_iim = f'intermediate-iim-{rate}-fill{int(fill_rate * 100)}.csv'
            start_time = time.time()
            process_and_fill_csv_iim(input_file_iim, output_file_iim, fill_rate)
            end_time = time.time()
            processing_time = end_time - start_time
            log_file.write(f"IIM processing {input_file_iim} with fill rate {fill_rate}: {processing_time:.4f} seconds.\n")

    # 均值填补：对所有中间文件进行均值填补
    for rate in Missing_rate:
        for fill_rate in Fill_rate:
            input_file_mean = f'intermediate-iim-{rate}-fill{int(fill_rate * 100)}.csv'
            output_file_mean = f'dirty-iim-mean-{rate}-fill{int(fill_rate * 100)}.csv'
            start_time = time.time()
            process_and_fill_csv_mean(input_file_mean, output_file_mean)
            end_time = time.time()
            processing_time = end_time - start_time
            log_file.write(f"Mean processing {input_file_mean}: {processing_time:.4f} seconds.\n")