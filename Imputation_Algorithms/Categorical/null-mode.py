import pandas as pd
import os

def process_and_fill(input_file, output_file, target_column, unrelated_column):
    df = pd.read_csv(input_file)
    if not df[target_column].empty:
        target_mode = df[target_column].mode()[0]
        df[target_column].fillna(target_mode, inplace=True)

    df.to_csv(output_file, index=False)
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
    output_path = os.path.join(base_path, dataset, "Imputation", "null-mode")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in Missing_rate:
        input_file = os.path.join(input_path, f'dirty-{rate}.csv')
        output_file = os.path.join(output_path, f'dirty-mode-{rate}')
        process_and_fill(input_file, output_file, target_column, unrelated_column)