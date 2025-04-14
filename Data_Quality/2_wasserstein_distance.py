import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

datasets = {
    "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"}
}
Imputation_Algorithms = ['mean', 'median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

def calculate_2wasserstein_distance(u, v):
    """
    计算2-Wasserstein距离。
    """
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    return np.sqrt(np.mean((u_sorted - v_sorted) ** 2))

for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]
    base_path = f"../Datasets/{dataset}"
    output_path = os.path.join(base_path, "Data_Quality", "2_wasserstein_distance")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_clean_file = os.path.join(base_path, "clean.csv")
    clean_data = pd.read_csv(input_clean_file)
    clean_target = clean_data[target_column].values
    mean = np.mean(clean_target)
    std = np.std(clean_target)
    clean_target_standardized = (clean_target - mean) / std

    results = []
    for rate in Missing_rate:
        for model in Imputation_Algorithms:
            input_dirty_file = os.path.join(base_path, "Imputation", f"null-{model}", f"dirty-{model}-{rate}.csv")
            dirty_data = pd.read_csv(input_dirty_file)
            dirty_target = dirty_data[target_column].values
            dirty_target_standardized = (dirty_target - mean) / std

            distance = calculate_2wasserstein_distance(clean_target_standardized, dirty_target_standardized)
            distance_less_than_0_05 = distance < 0.05

            results.append({
                'file': f"dirty-{model}-{rate}.csv",
                'model': model,
                'percentage': rate,
                '2-Wasserstein Distance': distance,
                'Distance_Less_Than_0_05': distance_less_than_0_05
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_path, "2_wasserstein_distance_results.csv"), index=False)
    print(f"2-Wasserstein Distance results saved to {os.path.join(output_path, '2_wasserstein_distance_results.csv')}")

    colors = plt.cm.viridis(np.linspace(0, 1, len(Imputation_Algorithms)))

    for percentage in Missing_rate:
        plt.figure(figsize=(12, 6))
        width = 0.4
        positions = np.arange(len(Imputation_Algorithms))

        for i, model in enumerate(Imputation_Algorithms):
            distances = results_df[(results_df['model'] == model) & (results_df['percentage'] == percentage)]['2-Wasserstein Distance']
            if not distances.empty:
                distance = distances.iloc[0]
                plt.bar(positions[i], distance, width=width, color=colors[i], label=model)

        plt.title(f'2-Wasserstein Distance for {percentage}% Missing Data in {dataset}')
        plt.xlabel('Model')
        plt.ylabel('2-Wasserstein Distance')
        plt.xticks(positions, Imputation_Algorithms, rotation=45, ha='right')
        plt.legend(title='Model', bbox_to_anchor=(1.16, 0.65), loc='upper right')
        plt.grid(False)

        output_fig_path = os.path.join(output_path, "fig")
        if not os.path.exists(output_fig_path):
            os.makedirs(output_fig_path)
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, f"2_wasserstein_distance_{percentage}_percent_barplot.png"))
        print(f"Bar plot saved to {output_fig_path}")