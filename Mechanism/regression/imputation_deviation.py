import pandas as pd
import numpy as np
import os

datasets = {
    "concrete": {"target_column": "concrete_compressive_strength", "nonnumerical_column": "None"},
    "CCPP": {"target_column": "PE", "nonnumerical_column": "None"},
    "AirfoilSelfNoise": {"target_column": "SSPL", "nonnumerical_column": "None"},
    "Abalone": {"target_column": "Rings", "nonnumerical_column": "None"},
    "ParisHousing": {"target_column": "price", "nonnumerical_column": "None"},

    "BostonHousePrice": {"target_column": "MEDV", "nonnumerical_column": "None"},
    "BostonHousePrice-history": {"target_column": "MEDV", "nonnumerical_column": "None"},
    "BostonHousePrice-test": {"target_column": "MEDV", "nonnumerical_column": "None"},
}
Imputation_Algorithms = ['mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

for dataset in datasets:
    target_column = datasets[dataset]["target_column"]
    nonnumerical_column = datasets[dataset]["nonnumerical_column"]
    base_path = "../../Datasets"
    output_path = os.path.join(base_path, dataset, "Mechanism", "regression", "imputation_deviation")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_clean_file = os.path.join(base_path, dataset, "clean.csv")
    clean_df = pd.read_csv(input_clean_file)

    results = pd.DataFrame(columns=['file', 'RMSE', 'MSE', 'MAE'])
    for model in Imputation_Algorithms:
        for rate in Missing_rate:
            input_dirty_file = f'dirty-{model}-{rate}.csv'
            try:
                input_dirty_path = os.path.join(base_path, dataset, "Imputation", f"null-{model}")
                dirty_df = pd.read_csv(os.path.join(input_dirty_path, input_dirty_file))
                rmse = np.sqrt(np.mean((dirty_df[target_column] - clean_df[target_column]) ** 2))
                mse = np.mean((dirty_df[target_column] - clean_df[target_column]) ** 2)
                mae = np.mean(np.abs(dirty_df[target_column] - clean_df[target_column]))

                new_row = pd.DataFrame({
                    'file': [input_dirty_file],
                    'RMSE': [rmse],
                    'MSE': [mse],
                    'MAE': [mae]
                })
                results = pd.concat([results, new_row], ignore_index=True)
            except Exception as e:
                print(f"Error processing file {input_dirty_file}: {e}")

    results.to_csv(os.path.join(output_path, 'imputation_deviation_results.csv'), index=False)
    print(f"results have saved to {os.path.join(output_path, 'imputation_deviation_results.csv')}")