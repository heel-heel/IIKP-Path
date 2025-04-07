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
    print(f'{output_file} has been saved.')


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    unrelated_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, unrelated_column)