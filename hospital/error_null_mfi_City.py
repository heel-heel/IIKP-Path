import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder

save_path='./null-City-mfi/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

read_path = './null-City/'

def process_and_fill_csv(file_name, output_file_name):
    df_copy = pd.read_csv(os.path.join(read_path,file_name))
    df = df_copy.replace("", pd.NA)
    df_drop = df.drop(columns=['ProviderNumber'])
    V2_index = df_drop.columns.get_loc('City')  # 获取V2列的索引


    missing_indices = df[df['City'].isnull()].index
    df_drop=df_drop.values
    M = np.ones_like(df_drop[:, V2_index])  # 创建掩码矩阵
    M[missing_indices] = 0

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_nan_values = df[column].dropna()
        df.loc[non_nan_values.index, column] = le.fit_transform(non_nan_values.astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    df_drop = df.drop(missing_indices)
    known_city_codes = df_drop['City'].values
    df_filled = df.drop(columns=['ProviderNumber'])
    # 将除 'city' 列外的其他列中的 NaN 值替换为 0
    columns_to_fill = df_filled.columns.difference(['City'])
    city_column_index = df_filled.columns.get_loc('City')
    df_filled[columns_to_fill] = df_filled[columns_to_fill].fillna(0)
    X=df_filled.values

    # 初始化参数
    p = 5  # 潜因子维度
    n, d = X.shape
    U = np.random.rand(n, p)  # 初始化U矩阵
    V = np.random.rand(d, p)  # 初始化V矩阵
    cmax = 1000  # 最大迭代次数
    threshold = 1 # 收敛阈值

    # 定义优化目标函数
    def objective(params):
        # 将一维数组重新构造为U和V矩阵
        U = params[:n * p].reshape(n, p)
        V = params[n * p:].reshape(d, p)
        X_pred = U @ V.T
        observed_diff = np.sum((X[M == 1] - X_pred[M == 1]) ** 2)
        return observed_diff

    # 定义优化过程
    def optimize(U, V):
        # 将U和V展平为一维数组
        params = np.concatenate([U.flatten(), V.flatten()])
        print(params)
        result = minimize(objective, params, method='BFGS', options={'maxiter': cmax})
        print(result)
        # 重新构造U和V矩阵
        U_opt = result.x[:n * p].reshape(n, p)
        V_opt = result.x[n * p:].reshape(d, p)
        return U_opt, V_opt

    # 迭代优化
    consecutive_count = 0  # 用于记录连续满足条件的次数
    previous_diffs = []  # 用于存储最近5次的avg_diff值

    for i in range(cmax):
        print("--------------------")
        print(f"echo:{i}")
        U, V = optimize(U, V)
        print('11111111111')
        X_pred = U @ V.T
        print('2222222222222')
        avg_diff = np.mean((X[M == 1] - X_pred[M == 1]) ** 2)
        print(avg_diff)

        # 更新previous_diffs列表，保持其长度为5
        previous_diffs.append(avg_diff)
        print(previous_diffs)
        if len(previous_diffs) > 1:
            previous_diffs.pop(0)  # 移除最早的值，保持列表长度为5

        # 检查连续5次的差值是否都小于1
        if len(previous_diffs) == 1:
            diffs = [abs(previous_diffs[j] - previous_diffs[j + 1]) for j in range(len(previous_diffs) - 1)]
            print(diffs)
            if all(diff < threshold for diff in diffs):
                print("连续5次的差值都小于1，停止迭代")
                break

    # 填充缺失值
    X_imputed = X.copy()
    X_imputed[missing_indices] = (U @ V.T)[missing_indices]

    for index in missing_indices:
        generated_value = X_imputed[index, city_column_index]
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
        city_name = label_encoders['City'].inverse_transform([int(generated_value)])[0]
        df_copy.at[index, 'City'] = city_name
        # print(city_name)

    #df_copy.iloc[:, V2_index+1] = X_imputed[:, V2_index]
    df_copy.to_csv(os.path.join(save_path,output_file_name), index=False)
    print(f"填充后的数据已导出到 {output_file_name}")

Missing_rate = [10, 30, 50, 70, 90]
for rate in Missing_rate:
    input_file = f'dirty-City-{rate}.csv'
    output_file = f'dirty-City-mfi-{rate}.csv'
    process_and_fill_csv(input_file, output_file)