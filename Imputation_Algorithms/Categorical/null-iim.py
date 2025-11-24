import pandas as pd
import numpy as np
import time
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder


def process_and_fill(input_file, output_file, target_column, unrelated_column):
    df_copy = pd.read_csv(input_file)
    if 'quality' in df_copy.columns:
        df_copy = pd.read_csv(input_file, dtype={'quality': 'object'})
    elif 'Type' in df_copy.columns:
        df_copy = pd.read_csv(input_file, dtype={'Type': 'object'})
    else:
        df_copy = pd.read_csv(input_file)
    df = df_copy.copy()

    print("Check for missing values...")
    columns_to_check = [col for col in df.columns if col != target_column]

    for col in columns_to_check:
        if df[col].isnull().any():
            missing_count = df[col].isnull().sum()
            print(f"The column '{col}' has {missing_count} missing values")

            if df[col].dtype in ['int64', 'float64']:
                # mean for numerical attributes
                fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
            else:
                # mode for categorical attributes
                if not df[col].mode().empty:
                    fill_value = df[col].mode()[0]
                    df[col] = df[col].fillna(fill_value)
                else:
                    # If the mode does not exist, use the first non-null value
                    fill_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else "Unknown"
                    df[col] = df[col].fillna(fill_value)

    target_observed_count = df[target_column].notnull().sum()
    missing_indices = df[df[target_column].isnull()].index

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_nan_values = df[column].dropna()
        df.loc[non_nan_values.index, column] = le.fit_transform(non_nan_values.astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    df_drop = df.drop(missing_indices)
    known_target_codes = df_drop[target_column].values

    if unrelated_column != "None":
        df_filled = df.drop(columns=[unrelated_column])
    else:
        df_filled = df
    columns_to_fill = df_filled.columns.difference([target_column])
    df_filled[columns_to_fill] = df_filled[columns_to_fill].fillna(0)

    # complete data and missing data
    complete_df = df.dropna(subset=[target_column])
    incomplete_df = df[df[target_column].isnull()]

    if len(complete_df) == 0:
        raise ValueError("No complete data available for learning")

    feature_columns = [col for col in df.columns if col != target_column]
    scaler = StandardScaler()
    X_complete_scaled = scaler.fit_transform(complete_df[feature_columns])
    X_incomplete_scaled = scaler.transform(incomplete_df[feature_columns])

    # learning stage
    def learning_phase(X_complete, y_complete, l_values):
        models_dict = {}
        n_complete = len(X_complete)

        for i in range(n_complete):
            current_sample = X_complete[i]
            current_value = y_complete[i]

            distances = np.linalg.norm(X_complete - current_sample, axis=1)
            sorted_indices = np.argsort(distances)
            sorted_indices = sorted_indices[sorted_indices != i]

            models_dict[i] = {}

            for l in l_values:
                if l >= len(sorted_indices):
                    continue

                neighbor_indices = sorted_indices[:l]

                if len(neighbor_indices) < 2:
                    if len(neighbor_indices) == 1:
                        models_dict[i][l] = {'type': 'direct', 'value': y_complete[neighbor_indices[0]]}
                    else:
                        models_dict[i][l] = {'type': 'direct', 'value': current_value}
                    continue

                X_neighbors = X_complete[neighbor_indices]
                y_neighbors = y_complete[neighbor_indices]

                try:
                    model = Ridge(alpha=1.0, random_state=42)
                    model.fit(X_neighbors, y_neighbors)

                    # evaluate
                    y_pred = model.predict(X_neighbors)
                    mse = mean_squared_error(y_neighbors, y_pred)

                    models_dict[i][l] = {
                        'type': 'ridge',
                        'model': model,
                        'mse': mse,
                        'neighbor_indices': neighbor_indices
                    }
                except:
                    models_dict[i][l] = {'type': 'mean', 'value': np.mean(y_neighbors)}

        return models_dict

    # choose the best model
    def select_optimal_models(models_dict, X_complete, y_complete, k_validation=5):
        optimal_models = {}
        n_complete = len(X_complete)

        for i in range(n_complete):
            current_sample = X_complete[i]
            current_value = y_complete[i]

            distances = np.linalg.norm(X_complete - current_sample, axis=1)
            sorted_indices = np.argsort(distances)
            validation_indices = sorted_indices[1:k_validation + 1]

            best_l = None
            best_error = float('inf')
            best_model_info = None

            for l, model_info in models_dict[i].items():
                total_error = 0
                valid_count = 0

                for j in validation_indices:
                    if j == i:
                        continue

                    validation_sample = X_complete[j]
                    true_value = y_complete[j]

                    if model_info['type'] == 'ridge':
                        pred_value = model_info['model'].predict(validation_sample.reshape(1, -1))[0]
                    elif model_info['type'] == 'mean':
                        pred_value = model_info['value']
                    elif model_info['type'] == 'direct':
                        pred_value = model_info['value']
                    else:
                        continue

                    error = (pred_value - true_value) ** 2
                    total_error += error
                    valid_count += 1

                if valid_count > 0:
                    avg_error = total_error / valid_count
                    if avg_error < best_error:
                        best_error = avg_error
                        best_l = l
                        best_model_info = model_info

            if best_model_info is None:
                if models_dict[i]:
                    first_l = list(models_dict[i].keys())[0]
                    best_model_info = models_dict[i][first_l]
                    print(f"Warning: Sample {i} did not find an optimal model, using the model with l={first_l}")
                else:
                    best_model_info = {'type': 'direct', 'value': current_value}
                    print(f"Warning: Sample {i} does not have any model, using default values")

            optimal_models[i] = best_model_info

        return optimal_models

    # imputation stage
    def imputation_phase(X_incomplete, X_complete, y_complete, optimal_models, k_imputation=5):
        predictions = []

        for incomplete_sample in X_incomplete:
            distances = np.linalg.norm(X_complete - incomplete_sample, axis=1)
            sorted_indices = np.argsort(distances)
            imputation_neighbors = sorted_indices[:k_imputation]

            candidate_predictions = []
            candidate_weights = []

            for neighbor_idx in imputation_neighbors:
                model_info = optimal_models[neighbor_idx]

                if model_info['type'] == 'ridge':
                    pred_value = model_info['model'].predict(incomplete_sample.reshape(1, -1))[0]
                elif model_info['type'] == 'mean':
                    pred_value = model_info['value']
                elif model_info['type'] == 'direct':
                    pred_value = model_info['value']
                else:
                    continue

                candidate_predictions.append(pred_value)
                candidate_weights.append(1.0)

            if len(candidate_predictions) > 1:
                consensus_scores = []
                for i, pred_i in enumerate(candidate_predictions):
                    consensus = 0
                    for j, pred_j in enumerate(candidate_predictions):
                        if i != j:
                            consensus += 1.0 / (1.0 + abs(pred_i - pred_j))
                    consensus_scores.append(consensus)

                total_consensus = sum(consensus_scores)
                if total_consensus > 0:
                    candidate_weights = [score / total_consensus for score in consensus_scores]

            final_prediction = np.average(candidate_predictions, weights=candidate_weights)
            predictions.append(final_prediction)

        return predictions

    y_complete = complete_df[target_column].values
    l_values = list(range(5, min(50, len(complete_df)), 5))
    if not l_values:
        l_values = [min(5, len(complete_df))]

    print(f"Learning individual models with l_values: {l_values}")
    models_dict = learning_phase(X_complete_scaled, y_complete, l_values)
    optimal_models = select_optimal_models(models_dict, X_complete_scaled, y_complete)

    if len(X_incomplete_scaled) > 0:
        predictions = imputation_phase(X_incomplete_scaled, X_complete_scaled, y_complete, optimal_models)
        for idx, pred_value in zip(incomplete_df.index, predictions):
            #df_copy.at[idx, target_column] = pred_value
            generated_value = pred_value
            min_code = np.min(known_target_codes)
            max_code = np.max(known_target_codes)

            if generated_value < min_code:
                generated_value = int(min_code)
            elif generated_value > max_code:
                generated_value = int(max_code)
            # else:
            #    distances = np.abs(known_city_codes - generated_value)
            #    closest_index = np.argmin(distances)
            #    generated_value = int(known_city_codes[closest_index])
            target_name = label_encoders[target_column].inverse_transform([int(generated_value)])[0]
            df_copy.at[idx, target_column] = target_name
        df_copy.to_csv(output_file, index=False)
        print(f'{output_file} has been saved.')


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    unrelated_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, unrelated_column)