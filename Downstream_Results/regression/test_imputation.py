def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

datasets = {
    "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"}
}
Imputation_Algorithms = ['mean', 'median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'rf', 'xgbi', 'gain', 'midae']
Missing_rate = [10, 30, 50, 70, 90]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def mlpc(X_train, X_test, y_train, y_test):
    model = MLPRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae

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

    for df in [rep_df, clean_df]:
        df[target] = df[target].astype(str).str.strip('%')
        df.loc[df[target] == 'empty', target] = 0
        df.loc[pd.isna(df[target]), target] = 0  
        df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0)
        for i in range(len(df[target])):
            try:
                df.loc[i, target] = float(df.loc[i, target]) 
            except:
                df[target][i] = 0

    df_encoded.rename(columns=sanitized_feature_names, inplace=True)
    train_indices, test_indices = train_test_split(range(len(df_encoded)), test_size=0.2, random_state=0)
    X_train = df_encoded.iloc[train_indices]
    y_train = rep_df[target].iloc[train_indices]
    X_test = df_encoded.iloc[test_indices]
    y_test = clean_df[target].iloc[test_indices]
    res_dict = {}
    mse, mae = mlpc(X_train, X_test, y_train, y_test)
    res_dict['mlpc'] = [mse, mae]
    return res_dict

if __name__ == "__main__":
    input_base_path = "../../Datasets"
    output_base_path = "../../Downstream_Results"
    for dataset, columns in datasets.items():
        print('-'*70)
        print(f"{dataset}:Processing...")
        target = columns["target_column"]
        input_clean_file = os.path.join(input_base_path, dataset, "clean.csv")
        clean_df = pd.read_csv(input_clean_file).astype(str)
        clean_df.fillna('nan', inplace=True)
        feature_schema = list(clean_df.columns)
        feature_schema.remove(target)

        results = []

        res_dict = testing_func(clean_df, clean_df, target, feature_schema)
        for algm in res_dict:
            results.append(["clean.csv", res_dict[algm][0], res_dict[algm][1]])
            print("'clean.csv' is ok." )
            print(f"{res_dict[algm][0]}, {res_dict[algm][1]}")

        for rate in Missing_rate:
            input_dirty_file = os.path.join(input_base_path, dataset, "null", f"dirty-{rate}.csv")
            dirty_df = pd.read_csv(input_dirty_file).astype(str)
            dirty_df.fillna('nan', inplace=True)
            res_dict = testing_func(dirty_df, clean_df, target, feature_schema)
            for algm in res_dict:
                results.append([f'dirty-{rate}.csv', res_dict[algm][0], res_dict[algm][1]])
                print(f"'dirty-{rate}.csv' is ok.")
                print(f"{res_dict[algm][0]}, {res_dict[algm][1]}")

        for model in Imputation_Algorithms:
            for rate in Missing_rate:
                input_imputed_file = os.path.join(input_base_path, dataset, "Imputation", f"null-{model}", f"dirty-{model}-{rate}.csv")
                imputed_df = pd.read_csv(input_imputed_file).astype(str)
                imputed_df.fillna('nan', inplace=True)
                res_dict = testing_func(imputed_df, clean_df, target, feature_schema)
                for algm in res_dict:
                    results.append([f'dirty-{model}-{rate}.csv', res_dict[algm][0], res_dict[algm][1]])
                    print(f"'dirty-{model}-{rate}.csv' is ok.")
                    print(f"{res_dict[algm][0]}, {res_dict[algm][1]}")

        output_results_file = os.path.join(output_base_path, "regression", dataset, f"mlp-imputation-results-{dataset}.csv")
        dir_path = os.path.dirname(output_results_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        results_df = pd.DataFrame(results, columns=["File Name", "MSE", "MAE"])
        results_df.to_csv(output_results_file, index=False)
