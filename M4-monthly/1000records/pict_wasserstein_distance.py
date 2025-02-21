import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

save_path = './sliced_wasserstein_distance/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def sliced_wasserstein_distance(X_original, X_imputed, num_directions=50, num_partitions=10):
    """
    计算Sliced Wasserstein distance。
    """
    n_samples, n_features = X_original.shape
    distances = []
    imputed_distances = []
    ratios = []  # 存储每次计算的比值

    for _ in range(num_partitions):
        indices = np.random.permutation(n_samples)
        half = n_samples // 2
        Ip = indices[:half]
        Jp = indices[half:]

        for _ in range(num_directions):
            direction = np.random.randn(n_features)
            direction /= np.linalg.norm(direction)

            original_proj = X_original @ direction
            imputed_proj = X_imputed @ direction

            distance = wasserstein_distance(original_proj[Ip], original_proj[Jp])
            imputed_distance = wasserstein_distance(original_proj[Ip], imputed_proj[Jp])

            distances.append(distance)
            imputed_distances.append(imputed_distance)

            if distance > 0:
                ratio = imputed_distance / distance
            else:
                ratio = imputed_distance

            ratios.append(ratio)  # 保留每次计算的比值

    avg_distance = np.mean(distances)
    avg_imputed_distance = np.mean(imputed_distances)
    avg_ratio = np.mean(ratios)
    std_distance = np.std(distances)
    std_imputed_distance = np.std(imputed_distances)
    std_ratio = np.std(ratios)

    return avg_distance, avg_imputed_distance, avg_ratio, std_distance, std_imputed_distance, std_ratio, ratios


def load_data_from_csv(original_path, imputed_path):
    """
    从CSV文件加载原始数据和填补后的数据。
    """
    df_original = pd.read_csv(original_path)
    df_imputed = pd.read_csv(imputed_path)

    X_original = df_original[['V2']].values
    X_imputed = df_imputed[['V2']].values

    return X_original, X_imputed


# 示例：加载数据
original_path = 'clean.csv'  # 替换为原始数据的CSV文件路径

# 定义模型名称和缺失率
model_names = ['mean', 'median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'randomforest', 'xgboost', 'gan', 'midae']
percentages = [10, 30, 50, 70, 90]

# 初始化存储每个缺失率的比值
model_ratios_by_percentage = {p: {model: [] for model in model_names} for p in percentages}

results = []

for percentage in percentages:
    csv_files = [f'dirty-{model}-{percentage}.csv' for model in model_names]

    for file in csv_files:
        model = file.split('-')[1]
        read_path = f'null-{model}'
        X_original, X_imputed = load_data_from_csv(original_path, os.path.join(read_path, file))

        mean = np.mean(X_original, axis=0)
        std = np.std(X_original, axis=0)

        if std == 0:
            raise ValueError("Standard deviation is zero. Cannot standardize the data.")

        X_original = (X_original - mean) / std
        X_imputed = (X_imputed - mean) / std

        avg_distance, avg_imputed_distance, avg_ratio, std_distance, std_imputed_distance, std_ratio, ratios = sliced_wasserstein_distance(X_original, X_imputed)

        print(f"Processing file: {file}")
        print(f"Average Distance: {avg_distance}")
        print(f"Average Imputed Distance: {avg_imputed_distance}")
        print(f"Average Ratio (Imputed Distance / Distance): {avg_ratio}")
        print(f"Standard Deviation of Distance: {std_distance}")
        print(f"Standard Deviation of Imputed Distance: {std_imputed_distance}")
        print(f"Standard Deviation of Ratio: {std_ratio}")
        print("-" * 50)

        results.append({
            'file': file,
            'avg_distance': avg_distance,
            'avg_imputed_distance': avg_imputed_distance,
            'avg_ratio': avg_ratio,
            'std_distance': std_distance,
            'std_imputed_distance': std_imputed_distance,
            'std_ratio': std_ratio
        })

        # 将当前缺失率的比值存储到对应的模型和缺失率分组中
        model_ratios_by_percentage[percentage][model].extend(ratios)

# 将结果保存到CSV文件
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_path, 'sliced_wasserstein_results.csv'), index=False)
print("Results saved to 'sliced_wasserstein_results.csv'")

# 绘制每个缺失率的箱线图
for percentage in percentages:
    plt.figure(figsize=(15, 6))
    data_to_plot = [model_ratios_by_percentage[percentage][model] for model in model_names]  # 按模型顺序排列数据
    plt.boxplot(data_to_plot, tick_labels=model_names, vert=True, patch_artist=True)
    plt.title(f'Boxplot of Ratios (Imputed Distance / Distance) for {percentage}% Missing Data')
    plt.ylabel('Ratio (Imputed Distance / Distance)')
    plt.xticks(rotation=45, ha='right')  # 旋转X轴标签以便更好地显示
    plt.grid(True)

    # 保存箱线图到指定路径
    boxplot_path = os.path.join(save_path, f'ratios_{percentage}_percent_boxplot.png')
    plt.tight_layout()  # 调整布局以防止标签被裁剪
    plt.savefig(boxplot_path)
    print(f"Boxplot saved to {boxplot_path}")