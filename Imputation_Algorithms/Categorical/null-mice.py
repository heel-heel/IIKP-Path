import pandas as pd
import numpy as np
import os
from statsmodels.imputation import mice
from sklearn.preprocessing import LabelEncoder


def process_and_fill(input_file, output_file, target_column, unrelated_column):
    df_copy = pd.read_csv(input_file)
    df = df_copy
    missing_indices = df[df[target_column].isnull()].index

    label_encoders = {}
    encoded_columns = []

    # 仅对非NaN值进行编码
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_nan_values = df[column].dropna()
        df.loc[non_nan_values.index, column] = le.fit_transform(non_nan_values.astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    if unrelated_column is not None:
        df_impute = df.drop(columns=[unrelated_column]).copy()
    else:
        df_impute = df.copy()
    np.random.seed(0)
    imp = mice.MICEData(df_impute)
    n_imputations = 100
    imputed_data = []

    for _ in range(n_imputations):
        imp.update(target_column)
        imputed_data.append(imp.data[target_column].copy())
    df_imputed = pd.concat(imputed_data, axis=1).mean(axis=1)

    for index in missing_indices:
        generated_value = df_imputed[index]
        city_name = label_encoders[target_column].inverse_transform([int(round(generated_value))])[0]
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
    output_path = os.path.join(base_path, dataset, "Imputation", "null-mice")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in Missing_rate:
        input_file = os.path.join(input_path, f'dirty-{rate}.csv')
        output_file = os.path.join(output_path, f'dirty-mice-{rate}')
        process_and_fill(input_file, output_file, target_column, unrelated_column)