import pandas as pd
import numpy as np
import os
from statsmodels.imputation import mice
from sklearn.preprocessing import LabelEncoder

save_path='./null-city-mice/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def process_and_fill_csv(file_name, output_file_name):
    df_copy = pd.read_csv(file_name)
    df = df_copy.replace("", pd.NA)
    missing_indices = df[df['city'].isnull()].index

    label_encoders = {}
    encoded_columns = []

    # 仅对非NaN值进行编码
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_nan_values = df[column].dropna()
        df.loc[non_nan_values.index, column] = le.fit_transform(non_nan_values.astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    df_impute = df.drop(columns=['id']).copy()
    np.random.seed(0)
    imp = mice.MICEData(df_impute)
    n_imputations = 100
    imputed_data = []

    for _ in range(n_imputations):
        imp.update('city')
        imputed_data.append(imp.data['city'].copy())
    df_imputed = pd.concat(imputed_data, axis=1).mean(axis=1)

    for index in missing_indices:
        generated_value = df_imputed[index]
        city_name = label_encoders['city'].inverse_transform([int(round(generated_value))])[0]
        df_copy.at[index, 'city'] = city_name

    df_copy.to_csv(os.path.join(save_path,output_file_name), index=False)


Missing_rate = [10, 30, 50, 70, 90]
for rate in Missing_rate:
    input_file = f'dirty-city-{rate}.csv'
    output_file = f'dirty-city-mice-{rate}.csv'
    process_and_fill_csv(input_file, output_file)