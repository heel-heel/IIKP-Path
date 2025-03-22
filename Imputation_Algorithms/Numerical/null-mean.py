import pandas as pd
import os

def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    data = pd.read_csv(input_file)
    global_mean = data[target_column].mean()
    data[target_column] = data[target_column].fillna(global_mean)
    data.to_csv(output_file, index=False)
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
    output_path = os.path.join(base_path, dataset, "Imputation", "null-mean")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in Missing_rate:
        input_file = os.path.join(input_path, f'dirty-{rate}.csv')
        output_file = os.path.join(output_path, f'dirty-mean-{rate}')
        process_and_fill(input_file, output_file, target_column, nonnumerical_column)