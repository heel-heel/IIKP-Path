import pandas as pd
import numpy as np
from scipy.stats import entropy
import os
import matplotlib.pyplot as plt

save_path = './kl_divergence/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def calculate_kl_divergence(clean_series, dirty_series, bins=20):
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

model_names = ['mean', 'median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'randomforest', 'xgboost', 'gan', 'midae']
percentages = [10, 30, 50, 70, 90]
csv_files = [f'dirty-{model}-{p}.csv' for model in model_names for p in percentages]

clean_data = pd.read_csv("clean.csv")
clean_v2 = clean_data['V2'].values

results = []
for file in csv_files:
    model = file.split('-')[1]
    percentage = int(file.split('-')[2].split('.')[0])
    dirty_path = os.path.join(f'./null-{model}', file)
    dirty_data = pd.read_csv(dirty_path)
    dirty_v2 = dirty_data['V2'].values

    kl_divergence = calculate_kl_divergence(clean_v2, dirty_v2)

    # 新增一列，判断KL散度是否小于0.05
    kl_divergence_less_than_0_05 = kl_divergence < 0.05

    results.append({
        'file': file,
        'model': model,
        'percentage': percentage,
        'KL_Divergence': kl_divergence,
        'KL_Divergence_Less_Than_0_05': kl_divergence_less_than_0_05  # 新增列
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_path, "kl_divergence_results.csv"), index=False)
print("KL Divergence results saved to 'kl_divergence_results.csv'")

# 绘制柱形图
colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))  # 为每个模型分配不同颜色

for percentage in percentages:
    plt.figure(figsize=(12, 6))
    width = 0.4
    positions = np.arange(len(model_names))  # 模型的位置

    for i, model in enumerate(model_names):
        kl_divs = results_df[(results_df['model'] == model) & (results_df['percentage'] == percentage)]['KL_Divergence']
        if not kl_divs.empty:
            kl_div = kl_divs.iloc[0]
            if np.isinf(kl_div):  # 检查是否为inf
                plt.text(positions[i], 0, 'inf', ha='center', va='bottom', fontsize=10, color='red')
            else:
                plt.bar(positions[i], kl_div, width=width, color=colors[i], label=model)

    plt.title(f'KL Divergence for {percentage}% Missing Data')
    plt.xlabel('Model')
    plt.ylabel('KL Divergence')
    plt.xticks(positions, model_names, rotation=45, ha='right')
    plt.legend(title='Model', bbox_to_anchor=(1.16, 0.65), loc='upper right')
    plt.grid(False)

    barplot_path = os.path.join(save_path, f'kl_divergence_{percentage}_percent_barplot.png')
    plt.tight_layout()
    plt.savefig(barplot_path)
    print(f"Bar plot saved to {barplot_path}")