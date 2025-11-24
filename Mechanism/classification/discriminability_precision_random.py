import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import StandardScaler

datasets = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    "Flights": {"target_column": "flight", "unrelated_column": None},
    "Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"},
    "RedWineQuality": {"target_column": "quality", "unrelated_column": "None"},
    "AvocadoRipeness": {"target_column": "ripeness", "unrelated_column": "None"}
}
label_cr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

base_path = "../../Datasets"

def calculate_consistent_rate_and_j_value(clean_df, filled_df, target_column, dataset):
    missing_positions = dirty_df[target_column].isnull()
    consistent_count = 0
    missing_total = missing_positions.sum()
    for row in range(missing_positions.shape[0]):
        if missing_positions[row]:
            if filled_df.at[row, target_column] == clean_df.at[row, target_column]:
                consistent_count += 1
    consistent_rate = consistent_count / missing_total if missing_total > 0 else 0

    features = filled_df.drop(columns=[target_column])
    target = filled_df[target_column]
    features_encoded = pd.get_dummies(features)

    if dataset == "Beers" or dataset == "Hospital":
        scaler = StandardScaler()
        numerical_features = features.select_dtypes(include=[np.number]).columns
        features_encoded[numerical_features] = scaler.fit_transform(features_encoded[numerical_features])

    unique_cities = target.unique()
    within_class_scatter = np.zeros((features_encoded.shape[1], features_encoded.shape[1]))
    between_class_scatter = np.zeros((features_encoded.shape[1], features_encoded.shape[1]))

    for city in unique_cities:
        class_data = features_encoded[target == city]
        class_mean = np.mean(class_data, axis=0)
        class_scatter = np.dot((class_data - class_mean).T, (class_data - class_mean))
        within_class_scatter += class_scatter

    overall_mean = np.mean(features_encoded, axis=0)

    for city in unique_cities:
        class_data = features_encoded[target == city]
        class_mean = np.mean(class_data, axis=0)
        class_size = len(class_data)
        between_class_scatter += class_size * np.outer((class_mean - overall_mean), (class_mean - overall_mean))

    within_class_distance = np.trace(within_class_scatter)
    between_class_distance = np.trace(between_class_scatter)
    J = between_class_distance / within_class_distance if within_class_distance > 0 else 0

    return consistent_rate, J, features_encoded, target


for dataset, columns in datasets.items():
    input_dirty_path = os.path.join(base_path, dataset, "null")
    input_clean_path = os.path.join(base_path, dataset)
    output_path = os.path.join(base_path, dataset, "Mechanism", "classification", "discriminability_precision_random")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_result = os.path.join(output_path, 'random-result.txt')
    with open(output_result, 'w') as f:
        f.write("CR_Target,CR_Actual,J_Actual\n")


    def save_results(cr_target, cr_actual, j_actual):
        with open(output_result, 'a') as f:
            f.write(f"{cr_target},{cr_actual},{j_actual}\n")


    clean_df = pd.read_csv(os.path.join(input_clean_path, "clean.csv"))
    dirty_df = pd.read_csv(os.path.join(input_dirty_path, "dirty-50.csv"))

    if dataset == "Beers":
        clean_df['ibu'] = clean_df['ibu'].fillna(clean_df['ibu'].mean())
        clean_df['ounces'] = clean_df['ounces'].fillna(clean_df['ounces'].mean())
        clean_df['abv'] = clean_df['abv'].fillna(clean_df['abv'].mean())
        dirty_df['ibu'] = dirty_df['ibu'].fillna(clean_df['ibu'].mean())
        dirty_df['ounces'] = dirty_df['ounces'].fillna(clean_df['ounces'].mean())
        dirty_df['abv'] = dirty_df['abv'].fillna(clean_df['abv'].mean())

    target_column = columns["target_column"]
    for cr_target in label_cr:
        print("-" * 70)
        print(f"Target CR: {cr_target}")

        for i in range(5):
            filled_df = dirty_df.copy()
            missing_indices = filled_df[filled_df[target_column].isnull()].index
            correct_size = int(len(missing_indices) * cr_target)
            correct_indices = np.random.choice(missing_indices, size=correct_size, replace=False)
            filled_df.loc[correct_indices, target_column] = clean_df.loc[correct_indices, target_column]
            error_indices = list(set(missing_indices) - set(correct_indices))
            # filled_df.loc[error_indices, target_column] = np.random.choice(clean_df[target_column].unique(), size=len(error_indices))
            for idx in error_indices:
                correct_value = clean_df.at[idx, target_column]
                possible_values = [v for v in clean_df[target_column].unique() if v != correct_value]
                if len(possible_values) == 0:
                    filled_value = correct_value
                else:
                    filled_value = np.random.choice(possible_values)
                filled_df.at[idx, target_column] = filled_value
            if 'quality' in target_column:
                filled_df[target_column] = filled_df[target_column].astype(int)

            cr_actual, j_actual, _, _ = calculate_consistent_rate_and_j_value(clean_df, filled_df, target_column,
                                                                              dataset)
            print(f"Run {i + 1}: CR_Actual={cr_actual}, J_Actual={j_actual}")

            output_file = os.path.join(output_path, f'random-{cr_target}-{i + 1}.csv')
            filled_df.to_csv(output_file, index=False)
            print(f"Saved {output_file} with CR_Actual={cr_actual}, J_Actual={j_actual}")
            save_results(cr_target, cr_actual, j_actual)
    print(f"{dataset}:All results have been saved.")