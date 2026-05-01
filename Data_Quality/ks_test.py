import pandas as pd
import numpy as np
import os
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import json

def evaluate_data_quality(datasets, Imputation_Algorithms, Missing_rate, Mechanism):
    colors = plt.cm.tab20(np.linspace(0, 1, len(Imputation_Algorithms)))
    markers = ['o', 'v', '+', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']
    color_dict = {model: color for model, color in zip(Imputation_Algorithms, colors)}

    for dataset, columns in datasets.items():
        target_column = columns["target_column"]
        nonnumerical_column = columns["nonnumerical_column"]

        base_path = f"../Datasets/{dataset}"
        output_path = os.path.join(base_path, "Data_Quality", "ks_test")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        clean_file_file = os.path.join(base_path, "clean.csv")
        clean_df = pd.read_csv(clean_file_file)
        clean_data = clean_df[target_column]

        for pattern in Mechanism:
            input_dirty_files = [f'dirty-{model}-{rate}.csv' for model in Imputation_Algorithms for rate in Missing_rate]
            ks_results = []
            for input_dirty_file in input_dirty_files:
                model = input_dirty_file.split('-')[1]
                rate = input_dirty_file.split('-')[2].split('.')[0]
                input_dirty_path = os.path.join(base_path, "Imputation", pattern, f"null-{model}")
                dirty_df = pd.read_csv(os.path.join(input_dirty_path, input_dirty_file))
                dirty_data = dirty_df[target_column]

                stat, p_value = ks_2samp(clean_data, dirty_data)
                ks_results.append((input_dirty_file, stat, p_value))

            results_df = pd.DataFrame(ks_results, columns=['file', 'Statistic', 'P-Value'])
            results_df['Same Distribution'] = results_df['P-Value'] > 0.05
            results_file_path = os.path.join(output_path, f'ks_test_results_{pattern}.csv')
            results_df.to_csv(results_file_path, index=False)
            print(f"KS Test results for {dataset} have been saved to '{results_file_path}'")

            plt.figure(figsize=(12, 6))
            sorted_dataset1 = np.sort(clean_data)
            yvals1 = np.arange(1, len(sorted_dataset1) + 1) / len(sorted_dataset1)
            plt.plot(sorted_dataset1, yvals1, label='clean', linewidth=3, color='black')

            # 绘制50%的其他数据集的CDF
            for input_dirty_file in input_dirty_files:
                if '50' in input_dirty_file:
                    model_name = input_dirty_file.split('-')[1]
                    input_dirty_path = os.path.join(base_path, "Imputation", pattern, f"null-{model_name}")
                    dirty_df = pd.read_csv(os.path.join(input_dirty_path, input_dirty_file))
                    dirty_data = dirty_df[target_column]
                    sorted_dataset = np.sort(dirty_data)
                    yvals = np.arange(1, len(sorted_dataset) + 1) / len(sorted_dataset)
                    plt.plot(sorted_dataset[sorted_dataset >= 0], yvals[sorted_dataset >= 0],
                             label=model_name, linewidth=3, color=color_dict[model_name])

            plt.legend(loc='lower right', fontsize='small')
            plt.title(f'Cumulative Distribution Functions (CDFs) for {dataset} under {pattern}', fontsize=16)
            plt.xlabel('Value', fontsize=14)
            plt.ylabel('Cumulative Probability', fontsize=14)

            output_fig_path = os.path.join(output_path, "fig")
            if not os.path.exists(output_fig_path):
                os.makedirs(output_fig_path)
            plt.tight_layout()
            plt.savefig(os.path.join(output_fig_path, f'cdf_plot_{pattern}.png'))
            plt.show()
            print(f"CDF plot for {dataset} has been saved to '{output_fig_path}'")

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