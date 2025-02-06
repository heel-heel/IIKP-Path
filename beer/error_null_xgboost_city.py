import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

save_path='./null-city-xgboost/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

read_path = './null-city/'

def process_and_fill_csv(file_name, output_file_name):
    df_copy = pd.read_csv(os.path.join(read_path,file_name))
    df = df_copy.replace("", pd.NA)
    missing_indices = df[df['city'].isnull()].index

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_nan_values = df[column].dropna()
        df.loc[non_nan_values.index, column] = le.fit_transform(non_nan_values.astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    df = df.drop(columns=['id']).fillna(0)
    #df_drop = df.drop(missing_indices)

    model = XGBRegressor()
    imputer = IterativeImputer(estimator=model, max_iter=30, random_state=0)
    features = df.columns
    data_imputed = imputer.fit_transform(df[features])
    data_imputed = pd.DataFrame(data_imputed, columns=features)

    #known_city_codes = df_drop['city'].values

    for index in missing_indices:
        generated_value = data_imputed.at[index, 'city']

        # 确保生成的数值在已知的城市编码范围内
        #min_code = np.min(known_city_codes)
        #max_code = np.max(known_city_codes)

        # 如果生成的数值超出范围，选择最接近的有效编码
        #if generated_value < min_code:
        #    generated_value = int(min_code)
        #elif generated_value > max_code:
        #    generated_value = int(max_code)
        #else:
        #    distances = np.abs(known_city_codes - generated_value)
        #    closest_index = np.argmin(distances)
        #    generated_value = int(known_city_codes[closest_index])
        city_name = label_encoders['city'].inverse_transform([int(generated_value)])[0]
        df_copy.at[index, 'city'] = city_name

    df_copy.to_csv(os.path.join(save_path,output_file_name), index=False)

# 主程序
Missing_rate = [10, 30, 50, 70, 90]
for rate in Missing_rate:
    input_file = f'dirty-city-{rate}.csv'
    output_file = f'dirty-city-xgboost-{rate}.csv'
    process_and_fill_csv(input_file, output_file)