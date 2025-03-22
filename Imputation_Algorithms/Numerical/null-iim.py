import pandas as pd
import numpy as np
import time
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    np.random.seed(42)
    df_copy = pd.read_csv(input_file)
    df = df_copy.drop(columns=nonnumerical_column)
    # 获取V2列上不为空的数量
    v2_observed_count = df[target_column].notnull().sum()
    print(v2_observed_count)
    missing_indices = df[df[target_column].isnull()].index
    #df_filled = df.fillna(df.mean())
    df_filled = df

    # 定义IIM填充函数
    def impute_iim(df_filled):
        df_imputed = df_filled.copy()
        feature = target_column  # 只针对目标列进行操作
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
        generated_value = df_imputed.at[index, target_column]
        df_copy.at[index, target_column] = generated_value
    df_copy.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')



datasets = {
    "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"}
}
Missing_rate = [10, 30, 50, 70, 90]
base_path = "../../Datasets"
for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]
    input_path = os.path.join(base_path, dataset, "null")
    output_path = os.path.join(base_path, dataset, "Imputation", "null-iim")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in Missing_rate:
        input_file = os.path.join(input_path, f'dirty-{rate}.csv')
        output_file = os.path.join(output_path, f'dirty-iim-{rate}')
        process_and_fill(input_file, output_file, target_column, nonnumerical_column)