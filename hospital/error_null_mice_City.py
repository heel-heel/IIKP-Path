import pandas as pd
import numpy as np
import os
from statsmodels.imputation import mice
from sklearn.preprocessing import LabelEncoder

save_path='./null-City-mice/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

read_path = './null-City/'

def process_and_fill_csv(file_name, output_file_name):
    df_copy = pd.read_csv(os.path.join(read_path,file_name))
    df = df_copy.replace("", pd.NA)
    missing_indices = df[df['City'].isnull()].index

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_nan_values = df[column].dropna()
        df.loc[non_nan_values.index, column] = le.fit_transform(non_nan_values.astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    #df = df.drop(columns=['id']).fillna(0)
    df_drop = df.drop(missing_indices)

    #df = pd.read_csv(file_name)
    df_impute = df.drop(columns=['ProviderNumber']).copy()
    np.random.seed(0)
    imp = mice.MICEData(df_impute)
    n_imputations = 50
    imputed_data = []

    for _ in range(n_imputations):
        imp.update('City')
        imputed_data.append(imp.data['City'].copy())
    df_imputed = pd.concat(imputed_data, axis=1).mean(axis=1)

    known_city_codes = df_drop['City'].values

    for index in missing_indices:
        generated_value = df_imputed[index]

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
        city_name = label_encoders['City'].inverse_transform([int(generated_value)])[0]
        df_copy.at[index, 'City'] = city_name

    df_copy.to_csv(os.path.join(save_path,output_file_name), index=False)

Missing_rate = [10,30,50,70,90]
#Missing_rate = [90]
for rate in Missing_rate:
    input_file = f'dirty-City-{rate}.csv'
    output_file = f'dirty-City-mice-{rate}.csv'
    process_and_fill_csv(input_file, output_file)