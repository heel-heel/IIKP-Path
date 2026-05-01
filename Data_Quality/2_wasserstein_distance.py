import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json

def evaluate_data_quality(datasets, Imputation_Algorithms, Missing_rate, Mechanism):
    def calculate_2wasserstein_distance(u, v):
        u_sorted = np.sort(u)
        v_sorted = np.sort(v)
        return np.sqrt(np.mean((u_sorted - v_sorted) ** 2))

    colors = plt.cm.tab20(np.linspace(0, 1, len(Imputation_Algorithms)))
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

        for pattern in Mechanism:
            results = []
            for rate in Missing_rate:
                for model in Imputation_Algorithms:
                    input_dirty_file = os.path.join(base_path, "Imputation", pattern, f"null-{model}", f"dirty-{model}-{rate}.csv")
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
            results_df.to_csv(os.path.join(output_path, f"2_wasserstein_distance_results_{pattern}.csv"), index=False)
            print(f"2-Wasserstein Distance results saved to {os.path.join(output_path, f'2_wasserstein_distance_results_{pattern}.csv')}")

            # figure
            for percentage in Missing_rate:
                plt.figure(figsize=(12, 6))
                width = 0.4
                positions = np.arange(len(Imputation_Algorithms))

                for i, model in enumerate(Imputation_Algorithms):
                    distances = results_df[(results_df['model'] == model) & (results_df['percentage'] == percentage)]['2-Wasserstein Distance']
                    if not distances.empty:
                        distance = distances.iloc[0]
                        #plt.bar(positions[i], distance, width=width, color=colors[i], label=model, hatch=markers[i])
                        plt.bar(positions[i], distance, width=width, color=colors[i], label=model)

                plt.title(f'2-Wasserstein Distance for {percentage}% Missing Data in {dataset}', fontsize=16)
                plt.xlabel('Imputation Algorithms', fontsize=14)
                plt.ylabel('2-Wasserstein Distance', fontsize=14)
                plt.xticks(positions, Imputation_Algorithms, rotation=45, ha='right', fontsize=10)
                plt.legend(title='Model', bbox_to_anchor=(1.11, 0.65), loc='upper right')
                plt.grid(False)

                output_fig_path = os.path.join(output_path, "fig", pattern)
                if not os.path.exists(output_fig_path):
                    os.makedirs(output_fig_path)
                plt.tight_layout()
                plt.savefig(os.path.join(output_fig_path, f"2_wasserstein_distance_{percentage}_percent_barplot.png"))
                print(f"Bar plot saved to {output_fig_path}")

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1]
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    datasets = config['datasets']
    Imputation_Algorithms = config['Imputation_Algorithms']
    Missing_rate = config['Missing_rate']
    Mechanism = config['Mechanism']
    evaluate_data_quality(datasets, Imputation_Algorithms, Missing_rate, Mechanism)