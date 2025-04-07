def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

datasets = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    "Flights": {"target_column": "flight", "unrelated_column": None},
    "Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"}
}
Imputation_Algorithms = ['mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'rf', 'xgbi', 'gain', 'midae']
Missing_rate = [10, 30, 50, 70, 90]

def mlpc(X_train, X_test, y_train, y_test):
    model = MLPClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1 = evaluate(y_test, y_pred)
    return precision, recall, f1

def testing_func(rep_df, clean_df, target, feature_schema):
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
    train_indices, test_indices = train_test_split(range(len(df_encoded)), test_size=0.2, random_state=0)
    X_train = df_encoded.iloc[train_indices]
    y_train = rep_df[target].iloc[train_indices]
    X_test = df_encoded.iloc[test_indices]
    y_test = clean_df[target].iloc[test_indices]
    res_dict = {}
    pre, rec, f1 = mlpc(X_train, X_test, y_train, y_test)
    res_dict['mlpc'] = [pre, rec, f1]
    return res_dict

def evaluate(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return precision, recall, f1

if __name__ == "__main__":
    input_base_path = "../../Datasets"
    output_base_path = "../../Downstream_Results"
    for dataset, columns in datasets.items():
        print('-'*70)
        print(f"{dataset}:Processing...")
        target = columns["target_column"]
        clean_path = os.path.join(input_base_path, dataset, 'clean.csv')
        clean_df = pd.read_csv(clean_path).astype(str)
        clean_df.fillna('nan', inplace=True)
        feature_schema = list(clean_df.columns)
        feature_schema.remove(target)

        # 初始化结果列表
        results = []

        # 处理清洁数据
        res_dict = testing_func(clean_df, clean_df, target, feature_schema)
        for algm in res_dict:
            results.append(["clean.csv", res_dict[algm][0], res_dict[algm][1], res_dict[algm][2]])
            print("'clean.csv' is ok.")
            print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}")

        # 处理脏数据
        for rate in Missing_rate:
            input_dirty_file = os.path.join(input_base_path, dataset, "null", f'dirty-{rate}.csv')
            dirty_df = pd.read_csv(input_dirty_file).astype(str)
            dirty_df.fillna('nan', inplace=True)
            res_dict = testing_func(dirty_df, clean_df, target, feature_schema)
            for algm in res_dict:
                results.append([f'dirty-{rate}.csv', res_dict[algm][0], res_dict[algm][1], res_dict[algm][2]])
                print(f"'dirty-{rate}.csv' is ok.")
                print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}")

        # 处理填补数据
        for model in Imputation_Algorithms:
            for rate in Missing_rate:
                imputed_path = os.path.join(input_base_path, dataset, "Imputation", f'null-{model}', f'dirty-{model}-{rate}.csv')
                imputed_df = pd.read_csv(imputed_path).astype(str)
                imputed_df.fillna('nan', inplace=True)
                res_dict = testing_func(imputed_df, clean_df, target, feature_schema)
                for algm in res_dict:
                    results.append([f'dirty-{model}-{rate}.csv', res_dict[algm][0], res_dict[algm][1], res_dict[algm][2]])
                    print(f"'dirty-{model}-{rate}.csv' is ok.")
                    print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}")

        output_results_file = os.path.join(output_base_path, "classification", dataset, f"mlp-imputation-results-{dataset}.csv")
        dir_path = os.path.dirname(output_results_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        results_df = pd.DataFrame(results, columns=["File Name", "Precision", "Recall", "F1 Score"])
        results_df.to_csv(output_results_file, index=False)