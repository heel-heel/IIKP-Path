import pandas as pd
import numpy as np
import os
import random

random.seed(42)
def insert_null(input_file, output_file, rate, target_column):
    df = pd.read_csv(input_file)
    num_missing = int(len(df) * rate / 100)
    random_indices = random.sample(range(len(df)), num_missing)
    df.loc[random_indices, target_column] = pd.NA

    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')

datasets = {
    #"M3-Yearly": "V2",
    "Glass": "Type",
    #"BostonHousePrice": "MEDV",

    #"M3-Yearly-history": "V2",
    #"M3-Yearly-test": "V2",
    "Glass-history": "Type",
    "Glass-test": "Type",
    #"BostonHousePrice-history": "MEDV",
    #"BostonHousePrice-test": "MEDV",
}
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
#Missing_rate = [80, 85, 90, 95]
base_path = "../Datasets"
for dataset, target_column in datasets.items():
    input_file = os.path.join(base_path, dataset, "clean.csv")
    output_path = os.path.join(base_path, dataset, "null")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in Missing_rate:
        output_file = os.path.join(output_path, f'dirty-{rate}.csv')
        insert_null(input_file, output_file, rate, target_column)