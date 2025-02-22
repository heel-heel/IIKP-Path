import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import os
import matplotlib.pyplot as plt

save_path = './2w_distance/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def calculate_2wasserstein_distance(clean_series, dirty_series):
    distance = wasserstein_distance(clean_series, dirty_series)
    return distance

model_names = ['mean', 'median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'randomforest', 'xgboost', 'gan', 'midae']
percentages = [10, 30, 50, 70, 90]
csv_files = [f'dirty-{model}-{p}.csv' for model in model_names for p in percentages]

clean_data = pd.read_csv("clean.csv")
clean_v2 = clean_data['V2'].values

mean = np.mean(clean_v2)
std = np.std(clean_v2)
clean_v2_standardized = (clean_v2 - mean) / std

results = []
for file in csv_files:
    model = file.split('-')[1]
    percentage = int(file.split('-')[2].split('.')[0])
    dirty_path = os.path.join(f'./null-{model}', file)
    dirty_data = pd.read_csv(dirty_path)
    dirty_v2 = dirty_data['V2'].values
    dirty_v2_standardized = (dirty_v2 - mean) / std

    distance = calculate_2wasserstein_distance(clean_v2_standardized, dirty_v2_standardized)

    # 新增一列，判断2-Wasserstein距离是否小于0.05
    distance_less_than_0_05 = distance < 0.05

    results.append({
        'file': file,
        'model': model,
        'percentage': percentage,
        '2-Wasserstein Distance': distance,
        'Distance_Less_Than_0_05': distance_less_than_0_05  # 新增列
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_path, "wasserstein_distances.csv"), index=False)
print("2-Wasserstein Distance results saved to 'wasserstein_distances.csv'")

# 绘制柱形图
colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))  # 为每个模型分配不同颜色

for percentage in percentages:
    plt.figure(figsize=(12, 6))
    width = 0.4
    positions = np.arange(len(model_names))  # 模型的位置

    for i, model in enumerate(model_names):
        # 获取当前模型和缺失率的数据
        distances = results_df[(results_df['model'] == model) & (results_df['percentage'] == percentage)]['2-Wasserstein Distance']
        if not distances.empty:
            distance = distances.iloc[0]
            plt.bar(positions[i], distance, width=width, color=colors[i], label=model)

    plt.title(f'2-Wasserstein Distance for {percentage}% Missing Data')
    plt.xlabel('Model')
    plt.ylabel('2-Wasserstein Distance')
    plt.xticks(positions, model_names, rotation=45, ha='right')
    plt.legend(title='Model', bbox_to_anchor=(1.16, 0.65), loc='upper right')
    plt.grid(False)

    # 保存柱形图到指定路径
    barplot_path = os.path.join(save_path, f'wasserstein_{percentage}_percent_barplot.png')
    plt.tight_layout()
    plt.savefig(barplot_path)
    print(f"Bar plot saved to {barplot_path}")