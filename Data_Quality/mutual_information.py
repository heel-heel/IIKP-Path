import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt
import json

def evaluate_data_quality(datasets, Imputation_Algorithms, Missing_rate):
    def calculate_mutual_information(original_df, imputed_df, column, discrete_features=None):
        if discrete_features is None:
            dtype = original_df[column].dtype
            is_discrete = isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_integer_dtype(dtype)
        else:
            is_discrete = discrete_features[original_df.columns.get_loc(column)]

        if column == 'Rings':
            mi = mutual_info_regression(original_df[[column]], imputed_df[column], discrete_features=[False], random_state=42)[0]
        elif is_discrete:
            mi = mutual_info_classif(original_df[[column]], imputed_df[column], discrete_features=[True], random_state=42)[0]
        else:
            mi = mutual_info_regression(original_df[[column]], imputed_df[column], discrete_features=[False], random_state=42)[0]

        return mi

    def process_and_calculate_mi(dataset, target_column, nonnumerical_column):
        base_path = "../Datasets"
        input_clean_file = os.path.join(base_path, dataset, "clean.csv")
        output_path = os.path.join(base_path, dataset, "Data_Quality", "mutual_information")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if nonnumerical_column != "None":
            clean_df = pd.read_csv(input_clean_file).drop(columns=[nonnumerical_column])
        else:
            clean_df = pd.read_csv(input_clean_file)

        results = []
        for model in Imputation_Algorithms:
            for rate in Missing_rate:
                input_dirty_file = os.path.join(base_path, dataset, "Imputation", f'null-{model}', f'dirty-{model}-{rate}.csv')
                if nonnumerical_column != "None":
                     dirty_df = pd.read_csv(input_dirty_file).drop(columns=[nonnumerical_column])
                else:
                    dirty_df = pd.read_csv(input_dirty_file)
                assert list(clean_df.columns) == list(dirty_df.columns), "原始数据和填补数据的特征列必须一致"
                mi_target = calculate_mutual_information(clean_df, dirty_df, column=target_column)
                results.append((f'dirty-{model}-{rate}.csv', mi_target))

        for file, mi in results:
            print(f"file: {file}, Mutual Information for {target_column}: {mi:.4f}")

        results_df = pd.DataFrame(results, columns=['file', 'Mutual_Information_target'])
        results_df.to_csv(os.path.join(output_path, 'mutual_information_results.csv'), index=False)
        fig, ax = plt.subplots(figsize=(15, 8))

        # 使用 plt.cm.tab20 生成颜色
        colors = plt.cm.tab20(np.linspace(0, 1, len(Imputation_Algorithms)))
        model_colors = {model: color for model, color in zip(Imputation_Algorithms, colors)}

        width = 0.32
        for idx, model in enumerate(Imputation_Algorithms):
            model_results = [mi for file, mi in results if model in file]
            ax.bar([p + idx * width for p in Missing_rate], model_results, width=width, label=model, color=model_colors[model])

        ax.set_xlabel("Missing rate", fontsize=14)
        ax.set_ylabel("Mutual Information", fontsize=14)
        ax.set_title(f"Mutual Information between Clean and Dirty Data for {dataset}", fontsize=16)
        ax.set_xticks([rate + (len(Imputation_Algorithms) - 1) * width / 2 for rate in Missing_rate])
        ax.set_xticklabels(Missing_rate)
        ax.legend()
        ax.grid(False)

        plt.tight_layout()
        output_fig_path = os.path.join(output_path, 'fig')
        if not os.path.exists(output_fig_path):
            os.makedirs(output_fig_path)
        plt.savefig(os.path.join(output_fig_path, 'mutual_information_plot.png'))
        plt.show()

    for dataset, columns in datasets.items():
        target_column = columns["target_column"]
        nonnumerical_column = columns["nonnumerical_column"]
        process_and_calculate_mi(dataset, target_column, nonnumerical_column)

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1]
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    datasets = config['datasets']
    Imputation_Algorithms = config['Imputation_Algorithms']
    Missing_rate = config['Missing_rate']
    evaluate_data_quality(datasets, Imputation_Algorithms, Missing_rate)