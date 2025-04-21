import pandas as pd
import numpy as np
from scipy.stats import entropy
import os
import matplotlib.pyplot as plt

datasets = {
    "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"}
}
Imputation_Algorithms = ['mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

# 颜色和标记设置
colors = plt.cm.tab20(np.linspace(0, 1, len(Imputation_Algorithms)))
#markers = ['o', 'v', '+', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']

def calculate_kl_divergence(clean_series, dirty_series, bins=20):
    """
    计算KL散度。
    """
    combined_data = np.concatenate((clean_series, dirty_series))
    hist, bin_edges = np.histogram(combined_data, bins=bins, density=True)
    clean_hist, _ = np.histogram(clean_series, bins=bin_edges, density=True)
    dirty_hist, _ = np.histogram(dirty_series, bins=bin_edges, density=True)
    clean_prob = clean_hist / np.sum(clean_hist)
    dirty_prob = dirty_hist / np.sum(dirty_hist)

    if np.any(clean_prob == 0):
        epsilon = 1e-9
        clean_prob = np.where(clean_prob == 0, epsilon, clean_prob)
    kl_div = entropy(dirty_prob, clean_prob)
    return kl_div

for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]
    base_path = f"../Datasets/{dataset}"
    output_path = os.path.join(base_path, "Data_Quality", "kl_divergence")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_clean_file = os.path.join(base_path, "clean.csv")
    clean_data = pd.read_csv(input_clean_file)
    clean_target = clean_data[target_column].values

    results = []
    for rate in Missing_rate:
        for model in Imputation_Algorithms:
            input_dirty_path = os.path.join(base_path, "Imputation", f"null-{model}")
            input_dirty_file = f"dirty-{model}-{rate}.csv"
            dirty_data = pd.read_csv(os.path.join(input_dirty_path, input_dirty_file))
            dirty_target = dirty_data[target_column].values
            kl_divergence = calculate_kl_divergence(clean_target, dirty_target)
            kl_divergence_less_than_0_05 = kl_divergence < 0.05

            results.append({
                'file': input_dirty_file,
                'model': model,
                'percentage': rate,
                'KL_Divergence': kl_divergence,
                'KL_Divergence_Less_Than_0_05': kl_divergence_less_than_0_05
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_path, "kl_divergence_results.csv"), index=False)
    print(f"KL Divergence results saved to {os.path.join(output_path, 'kl_divergence_results.csv')}")

    # 绘制柱形图
    #colors = plt.cm.viridis(np.linspace(0, 1, len(Imputation_Algorithms)))  # 为每个模型分配不同颜色

    for rate in Missing_rate:
        plt.figure(figsize=(12, 6))
        width = 0.4
        positions = np.arange(len(Imputation_Algorithms))

        for i, model in enumerate(Imputation_Algorithms):
            kl_divs = results_df[(results_df['model'] == model) & (results_df['percentage'] == rate)]['KL_Divergence']
            if not kl_divs.empty:
                kl_div = kl_divs.iloc[0]
                if np.isinf(kl_div):  # 检查是否为inf
                    plt.text(positions[i], 0, 'inf', ha='center', va='bottom', fontsize=10, color='red')
                else:
                    plt.bar(positions[i], kl_div, width=width, color=colors[i], label=model)

        plt.title(f'KL Divergence for {rate}% Missing Data in {dataset}',fontsize=16)
        plt.xlabel('Imputation Algorithms', fontsize=14)
        plt.ylabel('KL Divergence', fontsize=14)
        plt.xticks(positions, Imputation_Algorithms, rotation=45, ha='right',fontsize=10)
        plt.legend(title='Model', bbox_to_anchor=(1.11, 0.65), loc='upper right')
        plt.grid(False)

        output_fig_path = os.path.join(output_path, f"fig")
        if not os.path.exists(output_fig_path):
            os.makedirs(output_fig_path)
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, f'kl_divergence_{rate}_percent_barplot.png'))
        print(f"Bar plot saved to {os.path.join(output_fig_path, f'kl_divergence_{rate}_percent_barplot.png')}")