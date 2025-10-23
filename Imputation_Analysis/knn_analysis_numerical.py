import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import pairwise_distances

def calculate_knn_distances(input_file, target_column, nonnumerical_column, n_neighbors=5):
    df = pd.read_csv(input_file)
    if nonnumerical_column != "None":
        features = df.drop(columns=[nonnumerical_column])
    else:
        features = df

    features = features.drop(columns=[target_column])
    missing_indices = df[df[target_column].isnull()].index
    if len(missing_indices) == 0:
        print(f"在文件 {input_file} 中，目标列 '{target_column}' 没有缺失值")
        return 0

    numeric_data = features.select_dtypes(include=[np.number]).values
    distance_matrix = pairwise_distances(numeric_data, metric='euclidean')
    avg_distances = []

    for missing_idx in missing_indices:
        distances = distance_matrix[missing_idx]
        valid_indices = [i for i in range(len(distances))
                         if i != missing_idx and not pd.isna(df.iloc[i][target_column])]
        if len(valid_indices) < n_neighbors:
            print(f"警告: 对于索引 {missing_idx}，有效邻居数量不足 {n_neighbors}")
            continue
        valid_distances = distances[valid_indices]
        nearest_indices = np.argsort(valid_distances)[:n_neighbors]
        nearest_distances = valid_distances[nearest_indices]
        avg_distance = np.mean(nearest_distances)
        avg_distances.append(avg_distance)

    if len(avg_distances) == 0:
        print(f"在文件 {input_file} 中，无法为任何缺失点找到足够的邻居")
        return 0

    overall_avg_distance = np.mean(avg_distances)
    return overall_avg_distance

def process_datasets():
    # 数据集列表
    Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    datasets = {
        #"M4-Hourly": {"target_column": "V2", "nonnumerical_column": "V1"},
        "M4-Daily": {"target_column": "V2", "nonnumerical_column": "V1"},
        "M4-Weekly": {"target_column": "V2", "nonnumerical_column": "V1"},
        "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
        "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
        "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"},

        "concrete": {"target_column": "concrete_compressive_strength", "nonnumerical_column": "None"},
        "CCPP": {"target_column": "PE", "nonnumerical_column": "None"},
        "AirfoilSelfNoise": {"target_column": "SSPL", "nonnumerical_column": "None"},
        "Abalone": {"target_column": "Rings", "nonnumerical_column": "None"},
        "ParisHousing": {"target_column": "price", "nonnumerical_column": "None"},
    }
    input_base_path = os.path.join("../Datasets")

    #参数设置
    n_neighbors = 5

    results = []
    for dataset, columns in datasets.items():
        target_column = columns["target_column"]
        nonnumerical_column = columns["nonnumerical_column"]
        for rate in Missing_rate:
            input_path = os.path.join(input_base_path, dataset, "null")
            input_file = os.path.join(input_path, f"dirty-{rate}.csv")

            print(f"Processing: {input_file}")
            try:
                avg_distance = calculate_knn_distances(input_file, target_column, nonnumerical_column, n_neighbors)
                results.append({
                    'Dataset': input_file,
                    'Average_Distance': avg_distance,
                    'Missing_Count': pd.read_csv(input_file)[target_column].isnull().sum()
                })
                print(f"  - 平均距离: {avg_distance:.4f}")

            except Exception as e:
                print(f"  - 处理 {dataset} 时出错: {str(e)}")
                results.append({
                    'Dataset': dataset,
                    'Average_Distance': -1,  # 错误标记
                    'Missing_Count': -1
                })

    output_path = os.path.join("Results_numerical", "knn_analysis")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, "knn_analysis_results.txt")
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("KNN距离分析结果\n")
        f.write("=" * 50 + "\n")
        for result in results:
            f.write(f"数据集: {result['Dataset']}\n")
            if result['Average_Distance'] >= 0:
                f.write(f"缺失值数量: {result['Missing_Count']}\n")
                f.write(f"平均距离: {result['Average_Distance']:.6f}\n")
            else:
                f.write(f"处理错误\n")
            f.write("-" * 30 + "\n")
    print(f"{output_file} has been saved.")


if __name__ == "__main__":
    process_datasets()