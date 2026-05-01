import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import pairwise_distances

def calculate_hdi_distances(input_file, target_column, nonnumerical_column, n_neighbors=5):
    df = pd.read_csv(input_file)
    if nonnumerical_column != "None":
        features = df.drop(columns=[nonnumerical_column])
    else:
        features = df

    features = features.drop(columns=[target_column])
    missing_indices = df[df[target_column].isnull()].index
    if len(missing_indices) == 0:
        print(f"In the file {input_file}, the target column '{target_column}' has no missing values")
        return 0

    numeric_data = features.select_dtypes(include=[np.number]).values
    distance_matrix = pairwise_distances(numeric_data, metric='euclidean')
    avg_distances = []

    for missing_idx in missing_indices:
        distances = distance_matrix[missing_idx]
        valid_indices = [i for i in range(len(distances))
                         if i != missing_idx and not pd.isna(df.iloc[i][target_column])]
        if len(valid_indices) < n_neighbors:
            print(f"Warning: For index {missing_idx}, the number of valid neighbors is less than {n_neighbors}")
            continue
        valid_distances = distances[valid_indices]
        nearest_indices = np.argsort(valid_distances)[:n_neighbors]
        nearest_distances = valid_distances[nearest_indices]
        avg_distance = np.mean(nearest_distances)
        avg_distances.append(avg_distance)

    if len(avg_distances) == 0:
        print(f"In the file {input_file}, unable to find sufficient neighbors for any missing points")
        return 0

    overall_avg_distance = np.mean(avg_distances)
    return overall_avg_distance

def process_datasets():
    Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    datasets = {
        #"concrete": {"target_column": "concrete_compressive_strength", "nonnumerical_column": "None"},
        "CCPP": {"target_column": "PE", "nonnumerical_column": "None"},
        "AirfoilSelfNoise": {"target_column": "SSPL", "nonnumerical_column": "None"},
        "Abalone": {"target_column": "Rings", "nonnumerical_column": "None"},
        "ParisHousing": {"target_column": "price", "nonnumerical_column": "None"},
    }
    #Mechanism = ["MCAR", "MAR", "MNAR"]
    Mechanism = ["MCAR"]

    input_base_path = os.path.join("../Datasets")

    n_neighbors = 1
    results = []
    for dataset, columns in datasets.items():
        target_column = columns["target_column"]
        nonnumerical_column = columns["nonnumerical_column"]
        for pattern in Mechanism:
            for rate in Missing_rate:
                input_path = os.path.join(input_base_path, dataset, "null")
                input_file = os.path.join(input_path, pattern, f"dirty-{rate}.csv")

                print(f"Processing: {input_file}")
                try:
                    avg_distance = calculate_hdi_distances(input_file, target_column, nonnumerical_column, n_neighbors)
                    results.append({
                        'Dataset': input_file,
                        'Average_Distance': avg_distance,
                        'Missing_Count': pd.read_csv(input_file)[target_column].isnull().sum()
                    })
                    print(f"  - average distance: {avg_distance:.4f}")

                except Exception as e:
                    print(f"  - error when processing {dataset}: {str(e)}")
                    results.append({
                        'Dataset': dataset,
                        'Average_Distance': -1,
                        'Missing_Count': -1
                    })

    output_path = os.path.join("Results_numerical", "hdi_analysis")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, "hdi_analysis_results.txt")
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("HDI距离分析结果\n")
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