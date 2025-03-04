import pandas as pd
import numpy as np
import os
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

save_path = './null-iim/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

read_path = './null/'

def process_and_fill_csv(file_name, output_file_name):
    np.random.seed(42)
    df_copy = pd.read_csv(os.path.join(read_path,file_name))
    df = df_copy.drop(columns='V1')
    v2_observed_count = df['V2'].notnull().sum()
    print(v2_observed_count)
    missing_indices = df[df['V2'].isnull()].index
    df_filled = df

    # 定义IIM填充函数
    def impute_iim(df_filled):
        df_imputed = df_filled.copy()
        feature = 'V2'
        for i in range(len(df_filled)):
            if pd.isnull(df.iloc[i][feature]):
                print("---------------------")
                # 对每个K值训练回归模型
                models = []
                errors = []
                for k in range(5, v2_observed_count + 1,10):
                    # 找到最近的k个邻居
                    knn = KNeighborsRegressor(n_neighbors=k)
                    X_train = df_filled.dropna(subset=[feature]).drop(columns=[feature])
                    y_train = df_filled.dropna(subset=[feature])[feature]
                    knn.fit(X_train.values, y_train.values)
                    distances, indices = knn.kneighbors(df_filled.iloc[i].drop(feature).values.reshape(1, -1))

                    # 使用这些邻居作为样本训练回归模型
                    X_train_neighbor = X_train.iloc[indices[0]]
                    y_train_neighbor = y_train.iloc[indices[0]]
                    model = LinearRegression()
                    model.fit(X_train_neighbor, y_train_neighbor)
                    models.append(model)
                    #print(model.coef_)

                    # 使用邻居数据评估模型性能
                    y_pred_neighbor = model.predict(X_train_neighbor)
                    error = mean_squared_error(y_train_neighbor, y_pred_neighbor)
                    errors.append(error)

                # 选择最优模型
                best_model_idx = errors.index(min(errors))
                best_model = models[best_model_idx]

                # 使用最优模型预测缺失值
                df_imputed.at[i,feature] = best_model.predict(df_filled.iloc[i].drop(feature).to_frame().T)[0]
                print(df_imputed.at[i,feature])

        return df_imputed

    # 使用IIM方法填充缺失值
    df_imputed = impute_iim(df_filled)

    for index in missing_indices:
        generated_value = df_imputed.at[index, 'V2']
        df_copy.at[index, 'V2'] = generated_value
    df_copy.to_csv(os.path.join(save_path,output_file_name), index=False)
    print(f"{output_file_name}已保存到{save_path}")

log_file_path = os.path.join(save_path, "iim-time-log.txt")
with open(log_file_path, "w") as log_file:
    Missing_rate = [10, 30, 50, 70, 90]
    for rate in Missing_rate:
        input_file = f'dirty-{rate}.csv'
        output_file = f'dirty-iim-{rate}.csv'
        start_time = time.time()
        process_and_fill_csv(input_file, output_file)
        end_time = time.time()
        processing_time = end_time - start_time
        log_file.write(f"{input_file}:{processing_time:.4f} seconds.\n")