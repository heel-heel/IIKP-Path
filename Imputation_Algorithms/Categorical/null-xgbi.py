import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


def process_and_fill(input_file, output_file, target_column, unrelated_column):
    df_copy = pd.read_csv(input_file)
    df = df_copy
    missing_indices = df[df[target_column].isnull()].index

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_nan_values = df[column].dropna()
        df.loc[non_nan_values.index, column] = le.fit_transform(non_nan_values.astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    if unrelated_column is not None:
        df = df.drop(columns=[unrelated_column]).fillna(0)
    else:
        df = df.fillna(0)

    #df_drop = df.drop(missing_indices)

    model = XGBRegressor()
    imputer = IterativeImputer(estimator=model, max_iter=30, random_state=0)
    features = df.columns
    data_imputed = imputer.fit_transform(df[features])
    data_imputed = pd.DataFrame(data_imputed, columns=features)

    #known_city_codes = df_drop['city'].values

    for index in missing_indices:
        generated_value = data_imputed.at[index, target_column]

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
        city_name = label_encoders[target_column].inverse_transform([int(generated_value)])[0]
        df_copy.at[index, target_column] = city_name

    df_copy.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')

datasets = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    "Flights": {"target_column": "flight", "unrelated_column": None},
    "Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"}
}
Missing_rate = [10, 30, 50, 70, 90]
base_path = "../../Datasets"
for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    unrelated_column = columns["unrelated_column"]
    input_path = os.path.join(base_path, dataset, "null")
    output_path = os.path.join(base_path, dataset, "Imputation", "null-xgbi")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in Missing_rate:
        input_file = os.path.join(input_path, f'dirty-{rate}.csv')
        output_file = os.path.join(output_path, f'dirty-xgbi-{rate}')
        process_and_fill(input_file, output_file, target_column, unrelated_column)