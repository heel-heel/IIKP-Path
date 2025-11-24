import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

datasets = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    "Flights": {"target_column": "flight", "unrelated_column": None},
    "Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"},
    "RedWineQuality": {"target_column": "quality", "unrelated_column": "None"},
    "AvocadoRipeness": {"target_column": "ripeness", "unrelated_column": "None"}
}

for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    save_path = os.path.join("../../Datasets", dataset, "Mechanism", "classification", "discriminability_precision")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_file = os.path.join(save_path, 'log.txt')
    results_file = os.path.join(save_path, 'results.txt')
    with open(log_file, 'w') as f:
        f.write("")
    with open(results_file, 'w') as f:
        f.write("CR_Target,J_Target,CR_Actual,J_Actual\n")
    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    def save_results(cr_target, j_target, cr_actual, j_actual):
        with open(results_file, 'a') as f:
            f.write(f"{cr_target},{j_target},{cr_actual},{j_actual}\n")

    clean_df = pd.read_csv(os.path.join("../../Datasets", dataset, "clean.csv"))
    if 'quality' in clean_df.columns:
        clean_df = pd.read_csv(os.path.join("../../Datasets", dataset, "clean.csv"), dtype={'quality': 'object'})
    else:
        clean_df = pd.read_csv(os.path.join("../../Datasets", dataset, "clean.csv"))
    dirty_df = pd.read_csv(os.path.join("../../Datasets", dataset, "null", "dirty-50.csv"))
    if 'quality' in dirty_df.columns:
        dirty_df = pd.read_csv(os.path.join("../../Datasets", dataset, "null", "dirty-50.csv"), dtype={'quality': 'object'})
    else:
        dirty_df = pd.read_csv(os.path.join("../../Datasets", dataset, "null", "dirty-50.csv"))


    if dataset == "Beers":
        clean_df['ibu'] = clean_df['ibu'].fillna(clean_df['ibu'].mean())
        clean_df['ounces'] = clean_df['ounces'].fillna(clean_df['ounces'].mean())
        clean_df['abv'] = clean_df['abv'].fillna(clean_df['abv'].mean())

        dirty_df['ibu'] = dirty_df['ibu'].fillna(clean_df['ibu'].mean())
        dirty_df['ounces'] = dirty_df['ounces'].fillna(clean_df['ounces'].mean())
        dirty_df['abv'] = dirty_df['abv'].fillna(clean_df['abv'].mean())

    target_consistent_rates = np.arange(0.1, 1.0, 0.1).tolist()
    #j_value_targets = np.arange(0, 5.1, 0.1).tolist()
    j_value_targets = [0]
    max_iterations = 500
    max_cr_error = 0.01# The number of AvocadoRipeness entries is too small, requirements should be appropriately relaxed
    max_J_error = 0.05
    consecutive_count = 10  # Consecutive count
    #change_threshold = 1e-8  # Change threshold

    def find_intermediate_class(class_means, class_A, class_B):
        """Find the class closest to the average position of the two classes"""
        mean_AB = (class_means.loc[class_A] + class_means.loc[class_B]) / 2
        distances = np.linalg.norm(class_means - mean_AB, axis=1)
        return class_means.index[np.argmin(distances)]

    def get_boundary_samples(class_label, opposite_class, n, features_encoded, target):
        """Get the boundary samples closest to the opposing class within the specified class"""
        mask = (target == class_label)
        if sum(mask) == 0:
            return []
        distances = np.linalg.norm(features_encoded[mask] - class_means.loc[opposite_class], axis=1)
        return features_encoded[mask].index[np.argsort(distances)[:n]]

    def calculate_consistent_rate_and_j_value(clean_df, filled_df):
        # Calculate imputation accuracy
        missing_positions = dirty_df[target_column].isnull()
        consistent_count = 0
        missing_total = missing_positions.sum()
        for row in range(missing_positions.shape[0]):
            if missing_positions[row]:
                if filled_df.at[row, target_column] == clean_df.at[row, target_column]:
                    consistent_count += 1
        consistent_rate = consistent_count / missing_total if missing_total > 0 else 0

        # Calculate J value
        features = filled_df.drop(columns=[target_column])
        target = filled_df[target_column]
        features_encoded = pd.get_dummies(features)

        scaler = StandardScaler()
        numerical_features = features.select_dtypes(include=[np.number]).columns
        if len(numerical_features) > 0:
            features_encoded[numerical_features] = scaler.fit_transform(features_encoded[numerical_features])

        unique_labels = target.unique()
        within_class_scatter = np.zeros((features_encoded.shape[1], features_encoded.shape[1]))
        between_class_scatter = np.zeros((features_encoded.shape[1], features_encoded.shape[1]))

        for label in unique_labels:
            class_data = features_encoded[target == label]
            class_mean = np.mean(class_data, axis=0)
            class_scatter = np.dot((class_data - class_mean).T, (class_data - class_mean))
            within_class_scatter += class_scatter

        overall_mean = np.mean(features_encoded, axis=0)

        for label in unique_labels:
            class_data = features_encoded[target == label]
            class_mean = np.mean(class_data, axis=0)
            class_size = len(class_data)
            between_class_scatter += class_size * np.outer((class_mean - overall_mean), (class_mean - overall_mean))

        within_class_distance = np.trace(within_class_scatter)
        between_class_distance = np.trace(between_class_scatter)
        J = between_class_distance / within_class_distance if within_class_distance > 0 else 0

        return consistent_rate, J, features_encoded, target

    for cr_target in target_consistent_rates:
        log_message("-"*70)
        log_message("寻找上界......begin......")
        filled_df = dirty_df.copy()
        missing_indices = filled_df[filled_df[target_column].isnull()].index
        # Initial imputation: Generate correct/incorrect values according to the target CR
        correct_size = int(len(missing_indices) * cr_target)
        correct_indices = np.random.choice(missing_indices, size=correct_size, replace=False)
        filled_df.loc[correct_indices, target_column] = clean_df.loc[correct_indices, target_column]
        error_indices = list(set(missing_indices) - set(correct_indices))
        filled_df.loc[error_indices, target_column] = np.random.choice(clean_df[target_column].unique(), size=len(error_indices))
        max_recent_J_values = 0
        max_recent_J_counter = 0
        max_J_value = 0

        # Find the upper bound of the given CR
        while True:
            cr, J, features_encoded, target = calculate_consistent_rate_and_j_value(clean_df, filled_df)
            log_message(f"Now: J={J}, CR={cr}")
            cr_error = cr - cr_target
            #max_J_cr = cr
            #max_J_filled_df = filled_df.copy()

            # Update the current upper bound and record the number of times the upper bound appears
            if abs(cr_error) <= max_cr_error:
                if J > max_recent_J_values:
                    max_recent_J_values = J
                    max_J_filled_df = filled_df.copy()
                    max_J_cr = cr
                    max_recent_J_counter = 0
                else:
                    max_recent_J_counter = max_recent_J_counter + 1
                    if max_recent_J_counter >= consecutive_count:
                        max_J_value = max_recent_J_values
                        filled_file = os.path.join(save_path, f'filled-max-{cr_target:.1f}.csv')
                        max_J_filled_df.to_csv(filled_file, index=False)
                        log_message("找到上界！")
                        log_message(f"{filled_df}已保存到{save_path}")
                        save_results(cr_target, max_J_value, max_J_cr, max_J_value)
                        break


            # too low, improve CR
            if cr_error < -max_cr_error:
                log_message("\n微调CR......")
                class_means = features_encoded.groupby(target).mean()
                # Find all data points with incorrect imputation
                error_positions = [
                    idx for idx in missing_indices
                    if filled_df.at[idx, target_column] != clean_df.at[idx, target_column]
                ]
                if len(error_positions) == 0:
                    log_message("当前没有填补错误的单元格！")
                    break
                else:
                    # Calculate the distance from each incorrect data point to its correct category
                    distances_to_correct = []
                    for idx in error_positions:
                        correct_class = clean_df.at[idx, target_column]
                        current_features = features_encoded.loc[idx]

                        # Check if the correct category has appeared in filled_df
                        if correct_class not in filled_df[target_column].unique():
                            # If it has not appeared, directly set the distance to 0
                            distance = 0
                            log_message(f"Class '{correct_class}' not found in filled_df. Setting distance to 0.")
                        else:
                            # If it has appeared, calculate the actual distance
                            correct_class_mean = class_means.loc[correct_class]
                            distance = np.linalg.norm(current_features - correct_class_mean)

                        distances_to_correct.append((idx, distance))

                    n_corrections = min(len(error_positions), int(abs(cr_error) * len(missing_indices)))
                    log_message(f"Number of corrections to make: {n_corrections}")

                    distances_to_correct.sort(key=lambda x: x[1])  # Sort by distance in ascending order

                    # Modify the selected data points to the correct category
                    for idx, _ in distances_to_correct[:n_corrections]:
                        filled_df.at[idx, target_column] = clean_df.at[idx, target_column]
                        log_message(f"Corrected index {idx} to the correct class {clean_df.at[idx, target_column]}")

            # too high, reduce CR
            elif cr_error > max_cr_error:
                log_message("\n微调CR......")
                class_means = features_encoded.groupby(target).mean()
                # Find all data points with correct imputation
                correct_positions = [
                    idx for idx in missing_indices
                    if filled_df.at[idx, target_column] == clean_df.at[idx, target_column]
                ]
                if len(correct_positions) == 0:
                    log_message("No correct positions to corrupt. ERROR!")
                else:
                    # Calculate the distance from each correct data point to the center of its correct category
                    distances_to_corrupt = []
                    for idx in correct_positions:
                        correct_class = clean_df.at[idx, target_column]
                        current_features = features_encoded.loc[idx]
                        correct_class_mean = class_means.loc[correct_class]
                        distance = np.linalg.norm(current_features - correct_class_mean)
                        distances_to_corrupt.append((idx, distance))
                    # Calculate the number of samples that need adjustment
                    n_corruptions = min(len(correct_positions), int(abs(cr_error) * len(missing_indices)))
                    log_message(f"Number of corruptions to make: {n_corruptions}")

                    distances_to_corrupt.sort(key=lambda x: x[1], reverse=True)  # Sort by distance in descending order

                    # Modify the selected data points to the incorrect category
                    for idx, _ in distances_to_corrupt[:n_corruptions]:
                        correct_class = clean_df.at[idx, target_column]
                        current_features = features_encoded.loc[idx]
                        # Find the nearest/farthest incorrect category
                        distances_to_other_classes = []
                        for class_name, class_mean in class_means.iterrows():
                            if class_name != correct_class:
                                distance = np.linalg.norm(current_features - class_mean)
                                distances_to_other_classes.append((class_name, distance))
                        distances_to_other_classes.sort(key=lambda x: x[1])

                        # Select the target category
                        target_class = distances_to_other_classes[0][0]
                        filled_df.at[idx, target_column] = target_class
                        log_message(f"Corrupted index {idx} from '{correct_class}' to '{target_class}'")

            # If CR has been adjusted, the current CR and J value need to be recalculated
            if abs(cr_error) > max_cr_error:
                log_message("\n微调CR之后......")
                cr, J, features_encoded, target = calculate_consistent_rate_and_j_value(clean_df, filled_df)
                log_message(f"Now: J={J}, CR={cr}")
                cr_error = cr - cr_target
                if abs(cr_error) <= max_cr_error:
                    if J > max_recent_J_values:
                        max_recent_J_values = J
                        max_J_filled_df = filled_df.copy()
                        max_J_cr = cr
                        max_recent_J_counter = 0
                    else:
                        max_recent_J_counter = max_recent_J_counter + 1
                        if max_recent_J_counter >= consecutive_count:
                            max_J_value = max_recent_J_values
                            filled_file = os.path.join(save_path, f'filled-max-{cr_target:.1f}.csv')
                            max_J_filled_df.to_csv(filled_file, index=False)
                            log_message("找到上界！")
                            log_message(f"{filled_df}已保存到{save_path}")
                            save_results(cr_target,max_J_value,max_J_cr,max_J_value)
                            break

            # Increase the J value
            log_message("\n微调J Value......")
            class_means = features_encoded.groupby(target).mean()
            pairwise_dist = pairwise_distances(class_means)
            np.fill_diagonal(pairwise_dist, np.inf)

            # Dynamically calculate the adjustment magnitude
            # adjustment_rate = min(0.8, 0.2 * (1 + abs(J_error) * 5))
            adjustment_rate = 0.5

            # Calculate the distance from each data point to all class centers
            distances_to_means = pairwise_distances(features_encoded, class_means)
            # Find the nearest category and its distance for each data point
            nearest_classes = np.argmin(distances_to_means, axis=1)  # Index of the nearest category
            nearest_distances = np.min(distances_to_means, axis=1)  # Distance to the nearest category
            # Get the current category labels
            current_classes = target.values
            # Find points where the current category differs from the nearest category, and record their indices and distances
            reclassify_candidates = []
            for idx in range(len(features_encoded)):
                nearest_class_name = class_means.index[nearest_classes[idx]]  # Name of the nearest category
                current_class_name = current_classes[idx]
                # Ensure that category name comparisons are case-consistent
                if current_class_name.lower() != nearest_class_name.lower():
                    reclassify_candidates.append((idx, nearest_distances[idx]))

            # If there are no eligible points, skip directly
            if len(reclassify_candidates) <= 2:
                log_message("The number of points that need reclassfication is too small.")
                max_recent_J_counter = max_recent_J_counter + 1
                if max_recent_J_counter >= consecutive_count:
                    max_J_value = max_recent_J_values
                    filled_file = os.path.join(save_path, f'filled-max-{cr_target:.1f}.csv')
                    max_J_filled_df.to_csv(filled_file, index=False)
                    log_message("找到上界！")
                    log_message(f"{filled_df}已保存到{save_path}")
                    save_results(cr_target, max_J_value, max_J_cr, max_J_value)
                    break
            else:
                # Sort by distance in ascending order and select the n points with the smallest distance
                reclassify_candidates.sort(key=lambda x: x[1])  # Sort by distance
                # Dynamically adjust the range of random numbers based on the length of reclassify_candidates
                if len(reclassify_candidates) >= 10:
                    num_to_reclassify = random.randint(2, 10)
                else:
                    num_to_reclassify = random.randint(2, len(reclassify_candidates))
                # Separate correctly classified and incorrectly classified points
                correct_class_candidates = [(idx, dist) for idx, dist in reclassify_candidates if
                                            filled_df.at[idx, target_column] == clean_df.at[idx, target_column]]
                incorrect_class_candidates = [(idx, dist) for idx, dist in reclassify_candidates if
                                              filled_df.at[idx, target_column] != clean_df.at[idx, target_column]]

                # Ensure at least one correctly classified point and one incorrectly classified point are selected
                if correct_class_candidates:
                    correct_idx, _ = correct_class_candidates[0]  # Correctly classified point with the smallest distance
                else:
                    correct_idx = None

                if incorrect_class_candidates:
                    incorrect_idx, _ = incorrect_class_candidates[0]  # Incorrectly classified point with the smallest distance
                else:
                    incorrect_idx = None

                # If there are no correctly classified or incorrectly classified points, directly select the first num_to_reclassify points
                if correct_idx is None or incorrect_idx is None:
                    reclassify_indices = [idx for idx, _ in reclassify_candidates[:num_to_reclassify]]
                else:
                    # Merge these two points
                    selected_indices = [correct_idx, incorrect_idx]
                    # Ensure the number of selected points does not exceed num_to_reclassify
                    remaining_candidates = [idx for idx, _ in reclassify_candidates if idx not in selected_indices]
                    additional_indices = remaining_candidates[:num_to_reclassify-2]
                    selected_indices.extend(additional_indices)
                    reclassify_indices = selected_indices[:num_to_reclassify]

                log_message(f"Reclassifying {len(reclassify_indices)} points to their nearest classes")
                for idx in reclassify_indices:
                    current_class_name = current_classes[idx]
                    nearest_class_name = class_means.index[nearest_classes[idx]]  # Name of the nearest category

                    # Update categories
                    filled_df.at[idx, target_column] = nearest_class_name
                    log_message(f"Reclassified index {idx} from '{current_class_name}' to '{nearest_class_name}'")

        # Generate a series of files based on the given target values
        for j_target in j_value_targets:
            log_message("-" * 70)
            log_message("寻求目标J值......Begin......")
            log_message(f"Target: J={j_target}, CR={cr_target}")
            if (j_target > max_J_value):
                log_message("j_target > max_J_value，跳过")
                break
            filled_df = dirty_df.copy()
            missing_indices = filled_df[filled_df[target_column].isnull()].index

            # Initial imputation: Generate correct/incorrect values according to the target CR
            correct_size = int(len(missing_indices) * cr_target)
            correct_indices = np.random.choice(missing_indices, size=correct_size, replace=False)
            filled_df.loc[correct_indices, target_column] = clean_df.loc[correct_indices, target_column]
            error_indices = list(set(missing_indices) - set(correct_indices))
            filled_df.loc[error_indices, target_column] = np.random.choice(clean_df[target_column].unique(), size=len(error_indices))

            current_iter = 0
            min_recent_J_bias = 5
            min_J_bias_counter = 0
            min_J_bias_value = 0
            min_J_bias_cr = 0
            min_J_bias_filled_df = filled_df.copy()
            while current_iter < max_iterations:
                log_message("-"*70)
                cr, J, features_encoded, target = calculate_consistent_rate_and_j_value(clean_df, filled_df)
                log_message(f"Now: J={J}, CR={cr}")
                cr_error = cr - cr_target
                J_error = J - j_target

                # The current CR is within the acceptable range
                if abs(cr_error) <= max_cr_error:
                    if abs(J_error) < max_J_error:
                        min_J_bias_value = J
                        min_J_bias_cr = cr
                        min_J_bias_filled_df = filled_df.copy()
                        log_message("当前J值和CR都在可接受范围内！")
                        break
                    elif abs(J_error) < min_recent_J_bias:
                        min_recent_J_bias = abs(J_error)
                        min_J_bias_value = J
                        min_J_bias_cr = cr
                        min_J_bias_filled_df = filled_df.copy()
                        min_J_bias_counter = 0
                    else:
                        min_J_bias_counter = min_J_bias_counter + 1
                        if (min_J_bias_counter >= consecutive_count):
                            log_message(f"找到最接近目标J值：{min_J_bias_value}")
                            break

                # too low, increase CR
                if cr_error < -max_cr_error:
                    log_message("\n微调CR......")
                    class_means = features_encoded.groupby(target).mean()
                    # Find all data points with incorrect imputation
                    error_positions = [
                        idx for idx in missing_indices
                        if filled_df.at[idx, target_column] != clean_df.at[idx, target_column]
                    ]

                    if len(error_positions) == 0:
                        log_message("ERROR!")
                    else:
                        # Calculate the distance from each incorrect data point to its correct category
                        distances_to_correct = []
                        for idx in error_positions:
                            correct_class = clean_df.at[idx, target_column]
                            current_features = features_encoded.loc[idx]

                            # Check if the correct category has appeared in filled_df
                            if correct_class not in filled_df[target_column].unique():
                                # If it has not appeared, directly set the distance to 0
                                distance = 0
                                log_message(f"Class '{correct_class}' not found in filled_df. Setting distance to 0.")
                            else:
                                # If it has appeared, calculate the actual distance
                                correct_class_mean = class_means.loc[correct_class]
                                distance = np.linalg.norm(current_features - correct_class_mean)

                            distances_to_correct.append((idx, distance))

                        # Select data points based on the sign of J_error
                        n_corrections = min(len(error_positions), int(abs(cr_error) * len(missing_indices)))
                        log_message(f"Number of corrections to make: {n_corrections}")

                        if J_error < 0:  # Select the incorrect data point with the smallest distance
                            distances_to_correct.sort(key=lambda x: x[1])  # Sort by distance in ascending order
                        else:  # Select the incorrect data point with the largest distance
                            distances_to_correct.sort(key=lambda x: x[1], reverse=True)  # Sort by distance in descending order

                        # Modify the selected data points to the correct category
                        for idx, _ in distances_to_correct[:n_corrections]:
                            filled_df.at[idx, target_column] = clean_df.at[idx, target_column]
                            log_message(f"Corrected index {idx} to the correct class {clean_df.at[idx, target_column]}")

                # The current CR is too high and needs to be reduced
                elif cr_error > max_cr_error:
                    log_message("\n微调CR......")
                    class_means = features_encoded.groupby(target).mean()
                    # Find all data points with correct imputation
                    correct_positions = [
                        idx for idx in missing_indices
                        if filled_df.at[idx, target_column] == clean_df.at[idx, target_column]
                    ]
                    if len(correct_positions) == 0:
                        log_message("No correct positions to corrupt. ERROR!")
                    else:
                        # Calculate the distance from each correct data point to the center of its correct category
                        distances_to_corrupt = []
                        for idx in correct_positions:
                            correct_class = clean_df.at[idx, target_column]
                            current_features = features_encoded.loc[idx]
                            correct_class_mean = class_means.loc[correct_class]
                            distance = np.linalg.norm(current_features - correct_class_mean)
                            distances_to_corrupt.append((idx, distance))
                        # Calculate the number of samples that need adjustment
                        n_corruptions = min(len(correct_positions), int(abs(cr_error) * len(missing_indices)))
                        log_message(f"Number of corruptions to make: {n_corruptions}")

                        # Select data points based on the sign of J_error
                        if J_error < 0:  # Select the correct data point with the largest distance
                            distances_to_corrupt.sort(key=lambda x: x[1], reverse=True)  # Sort by distance in descending order
                        else:  # Select the correct data point with the smallest distance
                            distances_to_corrupt.sort(key=lambda x: x[1])  # Sort by distance in ascending order

                        # Modify the selected data points to the incorrect category
                        for idx, _ in distances_to_corrupt[:n_corruptions]:
                            correct_class = clean_df.at[idx, target_column]
                            current_features = features_encoded.loc[idx]
                            # Find the nearest/farthest incorrect category
                            distances_to_other_classes = []
                            for class_name, class_mean in class_means.iterrows():
                                if class_name != correct_class:
                                    distance = np.linalg.norm(current_features - class_mean)
                                    distances_to_other_classes.append((class_name, distance))

                            if J_error < 0:  # Modify to the nearest incorrect category
                                distances_to_other_classes.sort(key=lambda x: x[1])
                            else:  # Modify to the farthest incorrect category
                                distances_to_other_classes.sort(key=lambda x: x[1], reverse=True)

                            # Select the target category
                            target_class = distances_to_other_classes[0][0]
                            filled_df.at[idx, target_column] = target_class
                            log_message(f"Corrupted index {idx} from '{correct_class}' to '{target_class}'")

                # After CR fine-tuning, recalculation is required
                if abs(cr_error) > max_cr_error:
                    log_message("\n微调CR之后......")
                    cr, J, features_encoded, target = calculate_consistent_rate_and_j_value(clean_df, filled_df)
                    log_message(f"Now: J={J}, CR={cr}")
                    cr_error = cr - cr_target
                    J_error = J - j_target
                    if abs(cr_error) <= max_cr_error:
                        if abs(J_error) < max_J_error:
                            min_J_bias_value = J
                            min_J_bias_cr = cr
                            min_J_bias_filled_df = filled_df.copy()
                            log_message("当前J值和CR都在可接受范围内！")
                            break
                        elif abs(J_error) < min_recent_J_bias:
                            min_recent_J_bias = abs(J_error)
                            min_J_bias_value = J
                            min_J_bias_cr = cr
                            min_J_bias_filled_df = filled_df
                            min_J_bias_counter = 0
                        else:
                            min_J_bias_counter = min_J_bias_counter + 1
                            if (min_J_bias_counter >= consecutive_count):
                                log_message(f"找到最接近目标J值：{min_J_bias_value}")
                                break

                # Complete strategy modification for fine-tuning J values
                if abs(J_error) > max_J_error:
                    log_message("\n微调J Value......")
                    class_means = features_encoded.groupby(target).mean()
                    pairwise_dist = pairwise_distances(class_means)
                    np.fill_diagonal(pairwise_dist, np.inf)  # 屏蔽对角线

                    # Dynamically calculate the adjustment magnitude
                    #adjustment_rate = min(0.8, 0.2 * (1 + abs(J_error) * 5))
                    adjustment_rate = min(1, 0.5 * (1 + abs(J_error) * 5))

                    if J_error < 0:  # increase J value
                        log_message("Attempting to increase J value by reclassifying points to their nearest classes")

                        # Calculate the distance from each data point to all class centers
                        distances_to_means = pairwise_distances(features_encoded, class_means)

                        # Find the nearest category and its distance for each data point
                        nearest_classes = np.argmin(distances_to_means, axis=1)  # Index of the nearest category
                        nearest_distances = np.min(distances_to_means, axis=1)  # Distance to the nearest category

                        # Get the current category labels
                        current_classes = target.values

                        # Find points where the current category differs from the nearest category, and record their indices and distances
                        reclassify_candidates = []
                        for idx in range(len(features_encoded)):
                            nearest_class_name = class_means.index[nearest_classes[idx]]  # Name of the nearest category
                            current_class_name = current_classes[idx]

                            # Ensure that category name comparisons are case-consistent
                            if current_class_name.lower() != nearest_class_name.lower():
                                reclassify_candidates.append((idx, nearest_distances[idx]))

                        before_change = filled_df.copy()
                        # If there are no eligible points, skip directly
                        if len(reclassify_candidates) <= 2:
                            log_message("The number of points that need reclassfication is too small.")
                            break
                        else:
                            # Sort by distance in ascending order and select the n points with the smallest distance
                            reclassify_candidates.sort(key=lambda x: x[1])  # Sort by distance
                            # Dynamically adjust the range of random numbers based on the length of reclassify_candidates
                            if len(reclassify_candidates) >= 10:
                                num_to_reclassify = random.randint(2, 10)
                            else:
                                num_to_reclassify = random.randint(2, len(reclassify_candidates))
                            # Separate correctly classified and incorrectly classified points
                            correct_class_candidates = [(idx, dist) for idx, dist in reclassify_candidates if
                                                    filled_df.at[idx, target_column] == clean_df.at[idx, target_column]]
                            incorrect_class_candidates = [(idx, dist) for idx, dist in reclassify_candidates if
                                                    filled_df.at[idx, target_column] != clean_df.at[idx, target_column]]

                            # Ensure at least one correctly classified point and one incorrectly classified point are selected
                            if correct_class_candidates:
                                correct_idx, _ = correct_class_candidates[0]  # Correctly classified point with the smallest distance
                            else:
                                correct_idx = None

                            if incorrect_class_candidates:
                                incorrect_idx, _ = incorrect_class_candidates[0]  # Incorrectly classified point with the smallest distance
                            else:
                                incorrect_idx = None

                            # If there are no correctly classified or incorrectly classified points, directly select the first num_to_reclassify points
                            if correct_idx is None or incorrect_idx is None:
                                reclassify_indices = [idx for idx, _ in reclassify_candidates[:num_to_reclassify]]
                            else:
                                # Merge these two points
                                selected_indices = [correct_idx, incorrect_idx]
                                # Ensure the number of selected points does not exceed num_to_reclassify
                                remaining_candidates = [idx for idx, _ in reclassify_candidates if
                                                    idx not in selected_indices]
                                additional_indices = remaining_candidates[:num_to_reclassify - 2]
                                selected_indices.extend(additional_indices)
                                reclassify_indices = selected_indices[:num_to_reclassify]

                            log_message(f"Reclassifying {len(reclassify_indices)} points to their nearest classes")
                            for idx in reclassify_indices:
                                current_class_name = current_classes[idx]
                                nearest_class_name = class_means.index[nearest_classes[idx]]  # Name of the nearest category

                                # Update categories
                                filled_df.at[idx, target_column] = nearest_class_name
                                log_message(f"Reclassified index {idx} from '{current_class_name}' to '{nearest_class_name}'")

                    if J_error > 0:  # Need to decrease the J value
                        log_message("Attempting to decrease J value by reclassifying points to the farthest non-same classes")
                        # Calculate the distance from each data point to all class centers
                        distances_to_means = pairwise_distances(features_encoded, class_means)
                        # Find the nearest category and its distance for each data point
                        nearest_classes = np.argmin(distances_to_means, axis=1)  # Index of the nearest category
                        nearest_distances = np.min(distances_to_means, axis=1)  # Distance to the nearest category
                        # Get the current category labels
                        current_classes = target.values
                        # Find points where the current category is the same as the nearest category, and record their indices and distances
                        reclassify_candidates = []
                        for idx in range(len(features_encoded)):
                            nearest_class_name = class_means.index[nearest_classes[idx]]  # Name of the nearest category
                            current_class_name = current_classes[idx]
                            # Ensure that category name comparisons are case-consistent
                            if current_class_name.lower() == nearest_class_name.lower():
                                # Calculate the distance from this data point to all non-identical categories
                                distances_to_other_classes = []
                                for class_name, class_mean in class_means.iterrows():
                                    if class_name != current_class_name:
                                        distance = np.linalg.norm(features_encoded.iloc[idx] - class_mean)
                                        distances_to_other_classes.append((class_name, distance))
                                # Select the farthest non-identical category
                                if distances_to_other_classes:
                                    farthest_class = max(distances_to_other_classes, key=lambda x: x[1])[0]
                                    reclassify_candidates.append((idx, farthest_class))
                        # If there are no eligible points, skip directly
                        if len(reclassify_candidates) <= 2:
                            log_message("The number of points that need reclassfication is too small.")
                            break
                        else:
                            # Sort by distance in descending order and select the n points with the largest distance
                            reclassify_candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by distance
                            # Dynamically adjust the range of random numbers based on the length of reclassify_candidates
                            if len(reclassify_candidates) >= 10:
                                num_to_reclassify = random.randint(2, 10)
                            else:
                                num_to_reclassify = random.randint(2, len(reclassify_candidates))
                            # Separate correctly classified and incorrectly classified points
                            correct_class_candidates = [(idx, dist) for idx, dist in reclassify_candidates if
                                                   filled_df.at[idx, target_column] == clean_df.at[idx, target_column]]
                            incorrect_class_candidates = [(idx, dist) for idx, dist in reclassify_candidates if
                                                      filled_df.at[idx, target_column] != clean_df.at[idx, target_column]]

                            # Ensure at least one correctly classified point and one incorrectly classified point are selected
                            if correct_class_candidates:
                                correct_idx, _ = correct_class_candidates[0]  # Correctly classified point with the largest distance
                            else:
                                correct_idx = None

                            if incorrect_class_candidates:
                                 incorrect_idx, _ = incorrect_class_candidates[0]  # Incorrectly classified point with the largest distance
                            else:
                                 incorrect_idx = None

                            # If there are no correctly classified or incorrectly classified points, directly select the first num_to_reclassify points
                            if correct_idx is None or incorrect_idx is None:
                                reclassify_indices = [idx for idx, _ in reclassify_candidates[:num_to_reclassify]]
                            else:
                                # Merge these two points
                                selected_indices = [correct_idx, incorrect_idx]
                                # Ensure the number of selected points does not exceed num_to_reclassify
                                remaining_candidates = [idx for idx, _ in reclassify_candidates if
                                                    idx not in selected_indices]
                                additional_indices = remaining_candidates[:num_to_reclassify - 2]
                                selected_indices.extend(additional_indices)
                                reclassify_indices = selected_indices[:num_to_reclassify]

                            log_message(f"Reclassifying {len(reclassify_indices)} points to their farthest non-same classes")

                            for idx in reclassify_indices:
                                current_class_name = current_classes[idx]
                                farthest_class_name = reclassify_candidates[reclassify_indices.index(idx)][1]
                                # Update categories
                                filled_df.at[idx, target_column] = farthest_class_name
                                log_message(f"Reclassified index {idx} from '{current_class_name}' to '{farthest_class_name}'")

                current_iter += 1

            if(j_target <= max_J_value):
                # output
                filled_file = os.path.join(save_path, f'filled-min-{cr_target:.1f}.csv')
                min_J_bias_filled_df.to_csv(filled_file, index=False)
                log_message(f"{min_J_bias_filled_df}已保存到{save_path}")
                save_results(cr_target,j_target,min_J_bias_cr,min_J_bias_value)

        print("results.csv has been saved.")