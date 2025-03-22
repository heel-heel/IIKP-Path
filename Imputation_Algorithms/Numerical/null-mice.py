import pandas as pd
import numpy as np
import os
from statsmodels.imputation import mice

def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    df = pd.read_csv(input_file)
    df_impute = df.drop(columns=[nonnumerical_column]).copy()
    np.random.seed(0)
    imp = mice.MICEData(df_impute)
    n_imputations = 50
    imputed_data = []

    for _ in range(n_imputations):
        imp.update(target_column)
        imputed_data.append(imp.data[target_column].copy())
    df_imputed = pd.concat(imputed_data, axis=1).mean(axis=1)
    df.loc[df[target_column].isna(), target_column] = df_imputed
    df.to_csv(output_file, index=False)
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
    output_path = os.path.join(base_path, dataset, "Imputation", "null-mice")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in Missing_rate:
        input_file = os.path.join(input_path, f'dirty-{rate}.csv')
        output_file = os.path.join(output_path, f'dirty-mice-{rate}')
        process_and_fill(input_file, output_file, target_column, nonnumerical_column)