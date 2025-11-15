import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

datasets = {
    #"Beers": {"target_column": "city", "unrelated_column": "id"},
    #"Flights": {"target_column": "flight", "unrelated_column": None},
    #"Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"},
    #"RedWineQuality": {"target_column": "quality", "unrelated_column": "None"},
    #"AvocadoRipeness": {"target_column": "ripeness", "unrelated_column": "None"}

    #"Glass": {"target_column": "Type", "unrelated_column": "None"},
    "Glass-history": {"target_column": "Type", "unrelated_column": "None"},
    "Glass-test": {"target_column": "Type", "unrelated_column": "None"},
}
Imputation_Algorithms = ['mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


def calculate_class_discriminability(data, target_column, dataset_name):
    features = data.drop(columns=[target_column])
    target = data[target_column]
    features_encoded = pd.get_dummies(features)

    if dataset_name != "Flights":
        scaler = StandardScaler()
        numerical_features = features.select_dtypes(include=[np.number]).columns
        features_encoded[numerical_features] = scaler.fit_transform(features_encoded[numerical_features])

    unique_classes = target.unique()
    within_class_scatter = np.zeros((features_encoded.shape[1], features_encoded.shape[1]))

    for class_label in unique_classes:
        class_data = features_encoded[target == class_label]
        class_mean = np.mean(class_data, axis=0)
        class_scatter = np.dot((class_data - class_mean).T, (class_data - class_mean))
        within_class_scatter += class_scatter

    overall_mean = np.mean(features_encoded, axis=0)
    between_class_scatter = np.zeros((features_encoded.shape[1], features_encoded.shape[1]))
    for class_label in unique_classes:
        class_data = features_encoded[target == class_label]
        class_mean = np.mean(class_data, axis=0)
        class_size = len(class_data)
        between_class_scatter += class_size * np.outer((class_mean - overall_mean), (class_mean - overall_mean))

    within_class_distance = np.trace(within_class_scatter)
    between_class_distance = np.trace(between_class_scatter)
    J = between_class_distance / within_class_distance

    return {
        'Within-Class Distance': within_class_distance,
        'Between-Class Distance': between_class_distance,
        'J Value': J
    }


for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    unrelated_column = columns["unrelated_column"]
    base_path = "../../Datasets"
    output_path = os.path.join(base_path, dataset, "Mechanism", "classification")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results = []

    # First process the clean dataset
    input_clean_file = os.path.join(base_path, dataset, "clean.csv")
    print(f"Processing clean dataset for {dataset}...")
    try:
        clean_df = pd.read_csv(input_clean_file)

        if dataset == "Beers":
            clean_df['ibu'] = clean_df['ibu'].fillna(clean_df['ibu'].mean())
            clean_df['ounces'] = clean_df['ounces'].fillna(clean_df['ounces'].mean())
            clean_df['abv'] = clean_df['abv'].fillna(clean_df['abv'].mean())

        if dataset == "Hospital":
            clean_df['Score'] = clean_df['Score'].replace('empty', np.nan)


            def convert_to_decimal(x):
                try:
                    return float(x.strip('%')) / 100
                except (ValueError, AttributeError, TypeError):
                    return x


            clean_df['Score'] = clean_df['Score'].apply(convert_to_decimal)
            clean_df['Score'] = clean_df['Score'].fillna(clean_df['Score'].mean())

        clean_metrics = calculate_class_discriminability(clean_df, target_column, dataset)
        results.append({
            'file': 'clean.csv',
            'Within-Class Distance': clean_metrics['Within-Class Distance'],
            'Between-Class Distance': clean_metrics['Between-Class Distance'],
            'J Value': clean_metrics['J Value']
        })
        print(f"Clean dataset metrics calculated for {dataset}")
    except Exception as e:
        print(f"Error processing clean dataset for {dataset}: {e}")

    # Then process the imputed datasets
    for model in Imputation_Algorithms:
        for rate in Missing_rate:
            input_dirty_file = f'dirty-{model}-{rate}.csv'
            print(f"Processing...{input_dirty_file}")
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

                metrics = calculate_class_discriminability(dirty_data, target_column, dataset)

                print(f"正在计算：{input_dirty_file}")
                print(f"类内距离：{metrics['Within-Class Distance']}")
                print(f"类间距离：{metrics['Between-Class Distance']}")
                print(f"J值：{metrics['J Value']}")

                results.append({
                    'file': input_dirty_file,
                    'Within-Class Distance': metrics['Within-Class Distance'],
                    'Between-Class Distance': metrics['Between-Class Distance'],
                    'J Value': metrics['J Value']
                })
            except Exception as e:
                print(f"Error processing file {input_dirty_file}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_path, 'class_discriminability_results.csv'), index=False)
    print(f"评估结果已保存到 {os.path.join(output_path, 'class_discriminability_results.csv')}")