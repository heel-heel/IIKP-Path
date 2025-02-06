import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

save_path='./null-city-iim/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def process_and_fill_csv(file_name, output_file_name):
    df_copy = pd.read_csv(file_name)
    df = df_copy.replace("",pd.NA)
    v2_observed_count = df['city'].notnull().sum()
    missing_indices = df[df['city'].isnull()].index

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_nan_values = df[column].dropna()
        df.loc[non_nan_values.index, column] = le.fit_transform(non_nan_values.astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    df_drop = df.drop(missing_indices)
    known_city_codes = df_drop['city'].values
    df_filled = df.drop(columns=['id'])
    # 将除 'city' 列外的其他列中的 NaN 值替换为 0
    columns_to_fill = df_filled.columns.difference(['city'])
    df_filled[columns_to_fill] = df_filled[columns_to_fill].fillna(0)

    # 定义IIM填充函数
    def impute_iim(df_filled):
        df_imputed = df_filled
        feature = 'city'  # 只针对city列进行操作
        for i in range(len(df_filled)):
            if pd.isnull(df.iloc[i][feature]):
                print("---------------------")
                # 对每个k值训练回归模型
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
        generated_value = df_imputed.at[index, 'city']
        # 确保生成的数值在已知的城市编码范围内
        min_code = np.min(known_city_codes)
        max_code = np.max(known_city_codes)

        # 如果生成的数值超出范围，选择最接近的有效编码
        if generated_value < min_code:
            generated_value = int(min_code)
        elif generated_value > max_code:
            generated_value = int(max_code)
        #else:
        #    distances = np.abs(known_city_codes - generated_value)
        #    closest_index = np.argmin(distances)
        #    generated_value = int(known_city_codes[closest_index])
        city_name = label_encoders['city'].inverse_transform([int(generated_value)])[0]
        df_copy.at[index, 'city'] = city_name
        # print(city_name)
    df_copy.to_csv(os.path.join(save_path,output_file_name), index=False)

Missing_rate = [10, 30, 50, 70, 90]
for rate in Missing_rate:
    input_file = f'dirty-city-{rate}.csv'
    output_file = f'dirty-city-iim-{rate}.csv'
    process_and_fill_csv(input_file, output_file)