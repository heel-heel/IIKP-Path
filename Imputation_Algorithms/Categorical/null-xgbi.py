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

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    unrelated_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, unrelated_column)