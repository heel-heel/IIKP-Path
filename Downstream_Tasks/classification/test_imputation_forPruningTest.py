#def warn(*args, **kwargs):
#    pass
#import warnings
#warnings.warn = warn

import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
import numpy as np

# Define the parameter grid for random search
param_dist = {
    'hidden_layer_sizes': [(20,), (50,), (100,),
                           (50, 20), (100, 50), (100, 100),
                           (40, 20, 10), (50, 20, 10), (50, 20, 20), (50, 30, 10), (60, 30, 10),
                           (50, 20, 10, 10)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001],
    'learning_rate_init': [0.1, 0.01, 0.001, 0.0001, 0.00001],
    'max_iter': [200, 500, 1000, 2000, 3000, 5000, 8000],
    'early_stopping': [True, False]
}

# Datasets "Glass-history" and "Glass-classification_overview" will use the best parameters from "Glass"
#param_dist = {
#    'hidden_layer_sizes': [(50, 20)],
#    'activation': ['relu'],
#    'solver': ['adam'],
#    'alpha': [0.1],
#    'learning_rate_init': [0.0001],
#    'max_iter': [1000],
#    'early_stopping': [False]
#}

datasets = {
    "Glass": {"target_column": "Type", "unrelated_column": "None"},
    "Glass-history": {"target_column": "Type", "unrelated_column": "None"},
    "Glass-classification_overview": {"target_column": "Type", "unrelated_column": "None"},
}
Imputation_Algorithms = ['mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

def mlpc(X_train, X_test, y_train, y_test, best_params=None):
    if best_params:
        model = MLPClassifier(**best_params, random_state=42)
    else:
        model = MLPClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1 = evaluate(y_test, y_pred)
    return precision, recall, f1


def testing_func(rep_df, clean_df, target, feature_schema, best_params=None):
    feature_schema = [x.lower() for x in feature_schema]
    rep_df.columns = rep_df.columns.str.lower()
    clean_df.columns = clean_df.columns.str.lower()
    target = target.lower()
    df_encoded = pd.get_dummies(rep_df[feature_schema])

    sanitized_feature_names = {}
    for feature_names_str in df_encoded.columns:
        valid_chars = [char for char in feature_names_str if char not in ['[', ']', '<']]
        sanitized_feature_names[feature_names_str] = ''.join(valid_chars)

    df_encoded.rename(columns=sanitized_feature_names, inplace=True)
    train_indices, test_indices = train_test_split(range(len(df_encoded)), test_size=0.2, random_state=42)
    X_train = df_encoded.iloc[train_indices]
    y_train = rep_df[target].iloc[train_indices]
    X_test = df_encoded.iloc[test_indices]
    y_test = clean_df[target].iloc[test_indices]

    res_dict = {}
    pre, rec, f1 = mlpc(X_train, X_test, y_train, y_test, best_params)
    res_dict['mlpc'] = [pre, rec, f1]
    return res_dict


def evaluate(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return precision, recall, f1


def perform_random_search(X_train, y_train):
    mlp = MLPClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        mlp,
        param_distributions=param_dist,
        cv=5,  # 5-fold cross-validation
        random_state=42,
        n_jobs=-1,
        verbose=10,
        n_iter=50,
        scoring=make_scorer(f1_score, average='weighted')
    )
    random_search.fit(X_train, y_train)
    return random_search.best_params_


if __name__ == "__main__":
    input_base_path = "../../Datasets"
    output_base_path = "../../Downstream_Results"

    for dataset, columns in datasets.items():
        print('-' * 70)
        print(f"{dataset}:Processing...")
        target = columns["target_column"]
        clean_path = os.path.join(input_base_path, dataset, 'clean.csv')
        clean_df = pd.read_csv(clean_path).astype(str)
        if 'quality' in clean_df.columns:
            clean_df = pd.read_csv(clean_path, dtype={'quality': 'object'}).astype(str)
        elif 'Type' in clean_df.columns:
            clean_df = pd.read_csv(clean_path, dtype={'Type': 'object'}).astype(str)
        else:
            clean_df = pd.read_csv(clean_path).astype(str)
        clean_df.fillna('nan', inplace=True)
        feature_schema = list(clean_df.columns)
        feature_schema.remove(target)

        # Prepare data for random search
        clean_df_encoded = pd.get_dummies(clean_df[feature_schema])
        sanitized_feature_names = {}
        for feature_names_str in clean_df_encoded.columns:
            valid_chars = [char for char in feature_names_str if char not in ['[', ']', '<']]
            sanitized_feature_names[feature_names_str] = ''.join(valid_chars)
        clean_df_encoded.rename(columns=sanitized_feature_names, inplace=True)

        train_indices, test_indices = train_test_split(range(len(clean_df_encoded)), test_size=0.2, random_state=42)
        X_train_search = clean_df_encoded.iloc[train_indices]
        y_train_search = clean_df[target].iloc[train_indices]

        # Perform random search only on clean.csv
        print("Performing random search for best parameters...")
        best_params = perform_random_search(X_train_search, y_train_search)
        print(f"Best parameters found: {best_params}")

        # Initialize results list
        results = []

        # Process clean data with best parameters
        res_dict = testing_func(clean_df, clean_df, target, feature_schema, best_params)
        for algm in res_dict:
            clean_for_pg = res_dict[algm][2]
            results.append(["clean.csv", res_dict[algm][0], res_dict[algm][1], res_dict[algm][2], 0])
            print("'clean.csv' is ok.")
            print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}, 0")

        # Process dirty data with best parameters
        for rate in Missing_rate:
            input_dirty_file = os.path.join(input_base_path, dataset, "null", f'dirty-{rate}.csv')
            dirty_df = pd.read_csv(input_dirty_file).astype(str)
            if 'quality' in dirty_df.columns:
                dirty_df = pd.read_csv(input_dirty_file, dtype={'quality': 'object'}).astype(str)
            elif 'Type' in dirty_df.columns:
                dirty_df = pd.read_csv(input_dirty_file, dtype={'Type': 'object'}).astype(str)
            else:
                dirty_df = pd.read_csv(input_dirty_file).astype(str)
            dirty_df.fillna('nan', inplace=True)
            res_dict = testing_func(dirty_df, clean_df, target, feature_schema, best_params)
            for algm in res_dict:
                print(f"'dirty-{rate}.csv' is ok.")
                if res_dict[algm][2] > clean_for_pg:
                    results.append([f'dirty-{rate}.csv', res_dict[algm][0], res_dict[algm][1], res_dict[algm][2], 0])
                    print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}, 0")
                else:
                    results.append([f'dirty-{rate}.csv', res_dict[algm][0], res_dict[algm][1], res_dict[algm][2],
                                    (clean_for_pg - res_dict[algm][2]) / clean_for_pg])
                    print(
                        f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}, {(clean_for_pg - res_dict[algm][2]) / clean_for_pg}")

        # Process imputed data with best parameters
        for model in Imputation_Algorithms:
            for rate in Missing_rate:
                imputed_path = os.path.join(input_base_path, dataset, "Imputation", f'null-{model}',
                                            f'dirty-{model}-{rate}.csv')
                imputed_df = pd.read_csv(imputed_path).astype(str)
                if 'quality' in imputed_df.columns:
                    imputed_df = pd.read_csv(imputed_path, dtype={'quality': 'object'}).astype(str)
                elif 'Type' in imputed_df.columns:
                    imputed_df = pd.read_csv(imputed_path, dtype={'Type': 'object'}).astype(str)
                else:
                    imputed_df = pd.read_csv(imputed_path).astype(str)
                imputed_df.fillna('nan', inplace=True)
                res_dict = testing_func(imputed_df, clean_df, target, feature_schema, best_params)
                for algm in res_dict:
                    print(f"'dirty-{model}-{rate}.csv' is ok.")
                    if res_dict[algm][2] > clean_for_pg:
                        results.append(
                            [f'dirty-{model}-{rate}.csv', res_dict[algm][0], res_dict[algm][1], res_dict[algm][2], 0])
                        print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}, 0")
                    else:
                        results.append(
                            [f'dirty-{model}-{rate}.csv', res_dict[algm][0], res_dict[algm][1], res_dict[algm][2],
                             (clean_for_pg - res_dict[algm][2]) / clean_for_pg])
                        print(
                            f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}, {(clean_for_pg - res_dict[algm][2]) / clean_for_pg}")

        output_results_file = os.path.join(output_base_path, "classification", dataset, f"mlp-imputation-results-{dataset}.csv")
        dir_path = os.path.dirname(output_results_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        results_df = pd.DataFrame(results, columns=["File Name", "Precision", "Recall", "F1 Score", "PG(F1 Score)"])
        results_df.to_csv(output_results_file, index=False)