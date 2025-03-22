import pandas as pd
import numpy as np
import os
import random

def insert_null(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    num_missing = int(len(df) * rate / 100)
    random_indices = random.sample(range(len(df)), num_missing)
    df.loc[random_indices, target_column] = pd.NA

    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')



datasets = {
    "M4-Monthly": "V2",
    "M4-Quarterly": "V2",
    "M4-Yearly": "V2",
    "Beers": "city",
    "Flights": "flight",
    "Hospital": "City"
}
missing_rate = [10, 30, 50, 70, 90]
base_path = "../Datasets"

for dataset, target_column in datasets.items():
    input_file = os.path.join(base_path, dataset, "clean.csv")
    output_path = os.path.join(base_path, dataset, "null")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in missing_rate:
        output_file = os.path.join(output_path, f'dirty-{rate}.csv')
        insert_null(input_file, output_file, rate, target_column)