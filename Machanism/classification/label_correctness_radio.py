import pandas as pd
import os

datasets = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    "Flights": {"target_column": "flight", "unrelated_column": None},
    "Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"}
}
Imputation_Algorithms = ['mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'rf', 'xgbi', 'gain', 'midae']
Missing_rate = [10, 30, 50, 70, 90]

for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    unrelated_column = columns["unrelated_column"]
    base_path = "../../Datasets"
    output_path = os.path.join(base_path, dataset, "Machanism", "classification")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_clean_file = os.path.join(base_path, dataset, "clean.csv")
    clean_df = pd.read_csv(input_clean_file)

    results = []
    for model in Imputation_Algorithms:
        for rate in Missing_rate:
            input_dirty_file = f'dirty-{rate}.csv'
            input_filled_file = f'dirty-{model}-{rate}.csv'
            try:
                input_dirty_path = os.path.join(base_path, dataset, "null")
                dirty_df = pd.read_csv(os.path.join(input_dirty_path, input_dirty_file))
                input_filled_path = os.path.join(base_path, dataset, "Imputation", f"null-{model}")
                filled_df = pd.read_csv(os.path.join(input_filled_path, input_filled_file))
                consistent_count = 0
                missing_total = 0
                missing_positions = dirty_df[target_column].isnull()
                for row in range(missing_positions.shape[0]):
                    if missing_positions[row]:
                        missing_total += 1
                        if filled_df.at[row, target_column] == clean_df.at[row, target_column]:
                            consistent_count += 1
                consistent_rate = consistent_count / missing_total if missing_total > 0 else 0

                results.append({
                    'file': input_filled_file,
                    'Consistent Count': consistent_count,
                    'Missing Total': missing_total,
                    'Consistent Rate': consistent_rate
                })
            except Exception as e:
                print(f"Error processing file {input_filled_file}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_path, 'label_correstness_radio_results.csv'), index=False)
    print(f"评估结果已保存到 {os.path.join(output_path, 'label_correstness_radio_results.csv')}")