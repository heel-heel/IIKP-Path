import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

np.random.seed(42)
datasets = {
    #"M4-Hourly": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Daily": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Weekly": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"},

    "concrete": {"target_column": "concrete_compressive_strength", "nonnumerical_column": "None"},
    "CCPP": {"target_column": "PE", "nonnumerical_column": "None"},
    "AirfoilSelfNoise": {"target_column": "SSPL", "nonnumerical_column": "None"},
    "Abalone": {"target_column": "Rings", "nonnumerical_column": "None"},
    "ParisHousing": {"target_column": "price", "nonnumerical_column": "None"},
}
Imputation_Algorithms = ['mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
model_ratios_by_percentage = {rate: {model: [] for model in Imputation_Algorithms} for rate in Missing_rate}

def sliced_wasserstein_distance(X_clean, X_dirty, num_directions=50, num_partitions=10):
    """
    计算Sliced Wasserstein distance。
    """
    n_samples, n_features = X_clean.shape
    base_distances = []
    imputed_distances = []
    ratios = []

    for _ in range(num_partitions):
        indices = np.random.permutation(n_samples)
        if n_samples % 2 == 0:
            half = n_samples // 2
            Ip = indices[:half]
            Jp = indices[half:]
        else:
            half = n_samples // 2
            Ip = indices[:half]
            Jp = indices[half:half*2]


        for _ in range(num_directions):
            direction = np.random.randn(n_features)
            direction /= np.linalg.norm(direction)
            clean_proj = X_clean @ direction
            dirty_proj = X_dirty @ direction

            s = np.std(clean_proj[Ip])
            if s == 0:
                s = 1e-10
            clean_ip = (clean_proj[Ip] - np.mean(clean_proj[Ip])) / s
            clean_jp = (clean_proj[Jp] - np.mean(clean_proj[Ip])) / s
            dirty_jp = (dirty_proj[Jp] - np.mean(clean_proj[Ip])) / s

            def calc_w2(u, v):
                u_sorted = np.sort(u)
                v_sorted = np.sort(v)
                return np.sqrt(np.mean((u_sorted - v_sorted) ** 2))

            base_dist = calc_w2(clean_ip, clean_jp)
            imputed_dist = calc_w2(clean_ip, dirty_jp)

            base_distances.append(base_dist)
            imputed_distances.append(imputed_dist)
            if base_dist > 0:
                ratio = imputed_dist / base_dist
            else:
                ratio = imputed_dist
            ratios.append(ratio)

    avg_distance = np.mean(base_distances)
    avg_imputed_distance = np.mean(imputed_distances)
    avg_ratio = np.mean(ratios)
    std_distance = np.std(base_distances)
    std_imputed_distance = np.std(imputed_distances)
    std_ratio = np.std(ratios)
    return avg_distance, avg_imputed_distance, avg_ratio, std_distance, std_imputed_distance, std_ratio, ratios


for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]
    base_path = f"../Datasets/{dataset}"
    output_path = os.path.join(base_path, "Data_Quality", "sliced_wasserstein_distance")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results = []
    for rate in Missing_rate:
        for model in Imputation_Algorithms:
            input_clean_file = os.path.join(base_path, "clean.csv")
            input_dirty_path = os.path.join(base_path, "Imputation", f"null-{model}")
            input_dirty_file = f"dirty-{model}-{rate}.csv"

            clean_df = pd.read_csv(input_clean_file)
            dirty_df = pd.read_csv(os.path.join(input_dirty_path, input_dirty_file))
            if nonnumerical_column != "None":
                X_clean = clean_df.drop(columns=[nonnumerical_column]).values
                X_dirty = dirty_df.drop(columns=[nonnumerical_column]).values
            else:
                X_clean = clean_df.values
                X_dirty = dirty_df.values

            #mean = np.mean(X_clean, axis=0)
            #std = np.std(X_clean, axis=0)
            #if np.any(std == 0):
            #    raise ValueError("Standard deviation is zero. Cannot standardize the data.")
            #X_clean = (X_clean - mean) / std
            #X_dirty = (X_dirty - mean) / std
            avg_distance, avg_imputed_distance, avg_ratio, std_distance, std_imputed_distance, std_ratio, ratios = sliced_wasserstein_distance(X_clean, X_dirty)

            print(f"Processing file: {input_dirty_file}")
            print(f"Average Distance: {avg_distance}")
            print(f"Average Imputed Distance: {avg_imputed_distance}")
            print(f"Average Ratio (Imputed Distance / Distance): {avg_ratio}")
            print(f"Standard Deviation of Distance: {std_distance}")
            print(f"Standard Deviation of Imputed Distance: {std_imputed_distance}")
            print(f"Standard Deviation of Ratio: {std_ratio}")
            print("-" * 50)

            results.append({
                'file': input_dirty_file,
                'avg_distance': avg_distance,
                'avg_imputed_distance': avg_imputed_distance,
                'avg_ratio': avg_ratio,
                'std_distance': std_distance,
                'std_imputed_distance': std_imputed_distance,
                'std_ratio': std_ratio
            })

            model_ratios_by_percentage[rate][model].extend(ratios)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_path, 'sliced_wasserstein_distance_results.csv'), index=False)
    print(f"Results have saved to {os.path.join(output_path, 'sliced_wasserstein_distance_results.csv')}")

    for rate in Missing_rate:
        plt.figure(figsize=(12, 6))
        data_to_plot = [model_ratios_by_percentage[rate][model] for model in Imputation_Algorithms]  # 按模型顺序排列数据
        plt.boxplot(data_to_plot, labels=Imputation_Algorithms, vert=True, patch_artist=True)
        plt.title(f'Boxplot of Ratios (Imputed Distance / Distance) for {rate}% Missing Data in {dataset}', fontsize=16)
        plt.xlabel('Imputation Algorithms', fontsize=14)
        plt.ylabel('Ratio (Imputed Distance / Distance)', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        #plt.grid(True)

        output_fig_path = os.path.join(output_path, "fig")
        if not os.path.exists(output_fig_path):
            os.mkdir(output_fig_path)
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, f'ratios_{rate}_percent_boxplot.png'))
        print(f"Boxplot has saved to {output_fig_path}")