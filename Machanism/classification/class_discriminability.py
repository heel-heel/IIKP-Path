import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

datasets = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    "Flights": {"target_column": "flight", "unrelated_column": None},
    "Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"}
}
Imputation_Algorithms = ['mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'rf', 'xgbi', 'gain', 'midae']
Missing_rate = [10, 30, 50, 70, 90]

for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    unrelated_column = columns["unrelated_column"]
    base_path = "../../Datasets"
    output_path = os.path.join(base_path, dataset, "Machanism", "classification")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results = []
    input_clean_file = os.path.join(base_path, dataset, "clean.csv")
    clean_df = pd.read_csv(input_clean_file)

    for model in Imputation_Algorithms:
        for rate in Missing_rate:
            input_dirty_file = f'dirty-{model}-{rate}.csv'
            try:
                input_dirty_path = os.path.join(base_path, dataset, "Imputation", f"null-{model}")
                dirty_data = pd.read_csv(os.path.join(input_dirty_path, input_dirty_file))

                if dataset == "Beers":
                    dirty_data['ibu'] = dirty_data['ibu'].fillna(dirty_data['ibu'].mean())
                    dirty_data['ounces'] = dirty_data['ounces'].fillna(dirty_data['ounces'].mean())
                    dirty_data['abv'] = dirty_data['abv'].fillna(dirty_data['abv'].mean())

                if dataset == "Hospital":
                    dirty_data['Score'] = dirty_data['Score'].replace('empty', np.nan)
                    def convert_to_decimal(x):
                        try:
                            return float(x.strip('%')) / 100
                        except (ValueError, AttributeError, TypeError):
                            return x
                    dirty_data['Score'] = dirty_data['Score'].apply(convert_to_decimal)
                    dirty_data['Score'] = dirty_data['Score'].fillna(dirty_data['Score'].mean())

                features = dirty_data.drop(columns=[target_column])
                target = dirty_data[target_column]
                features_encoded = pd.get_dummies(features)

                if dataset != "Flights":
                    scaler = StandardScaler()
                    numerical_features = features.select_dtypes(include=[np.number]).columns
                    features_encoded[numerical_features] = scaler.fit_transform(features_encoded[numerical_features])

                unique_cities = target.unique()
                within_class_scatter = np.zeros((features_encoded.shape[1], features_encoded.shape[1]))

                for city in unique_cities:
                    class_data = features_encoded[target == city]
                    class_mean = np.mean(class_data, axis=0)
                    class_scatter = np.dot((class_data - class_mean).T, (class_data - class_mean))
                    within_class_scatter += class_scatter

                overall_mean = np.mean(features_encoded, axis=0)
                between_class_scatter = np.zeros((features_encoded.shape[1], features_encoded.shape[1]))
                for city in unique_cities:
                    class_data = features_encoded[target == city]
                    class_mean = np.mean(class_data, axis=0)
                    class_size = len(class_data)
                    between_class_scatter += class_size * np.outer((class_mean - overall_mean), (class_mean - overall_mean))

                within_class_distance = np.trace(within_class_scatter)
                between_class_distance = np.trace(between_class_scatter)
                J = between_class_distance / within_class_distance

                print(f"正在计算：{input_dirty_file}")
                print(f"类内距离：{within_class_distance}")
                print(f"类间距离：{between_class_distance}")
                print(f"J值：{J}")

                results.append({
                    'file': input_dirty_file,
                    'Within-Class Distance': within_class_distance,
                    'Between-Class Distance': between_class_distance,
                    'J Value': J
                })
            except Exception as e:
                print(f"Error processing file {input_dirty_file}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_path, 'class_discriminability_results.csv'), index=False)
    print(f"评估结果已保存到 {os.path.join(output_path, 'class_discriminability_results.csv')}")