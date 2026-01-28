# def warn(*args, **kwargs):
#    pass
# import warnings
# warnings.warn = warn

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor

datasets = {
    "BostonHousePrice": {"target_column": "MEDV", "nonnumerical_column": "None"},
    "BostonHousePrice-history": {"target_column": "MEDV", "nonnumerical_column": "None"},
    "BostonHousePrice-classification_overview": {"target_column": "MEDV", "nonnumerical_column": "None"},
}
Imputation_Algorithms = ['mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


def get_best_mlp_params(X_train, y_train):
    # Define the parameter grid for random search
    param_grid = {
        'hidden_layer_sizes': [(50,), (80,), (100,), (120,), (150,),
                               (50, 10), (50, 50), (100, 50),
                               (40, 20, 10), (50, 20, 10), (50, 20, 20), (50, 30, 10), (60, 30, 10),
                               (50, 20, 10, 10)
                               ],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'learning_rate_init': [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'max_iter': [200, 500, 1000, 2000, 3000, 5000, 8000],
        'early_stopping': [True, False]
     }

    # Datasets "BostonHousePrice-history" and "BostonHousePrice-classification_overview" will use the best parameters from "BostonHousePrice"
#    param_grid = {
#        'hidden_layer_sizes': [(120,)],
#        'activation': ['relu'],
#        'solver': ['sgd'],
#        'alpha': [0.0001],
#        'learning_rate_init': [0.0001],
#        'max_iter': [8000],
#        'early_stopping': [False]
#    }

    mlp = MLPRegressor(random_state=42)
    grid_search = RandomizedSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=10, n_iter=50)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_


def mlpc(X_train, X_test, y_train, y_test, best_params=None):
    """使用最优参数训练MLP模型"""
    if best_params is None:
        model = MLPRegressor(random_state=42)
    else:
        model = MLPRegressor(**best_params, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse, mae = evaluate(y_test, y_pred)
    return mse, mae


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
    mse, mae = mlpc(X_train, X_test, y_train, y_test, best_params)
    res_dict['mlpc'] = [mse, mae]
    return res_dict


def evaluate(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae


if __name__ == "__main__":
    input_base_path = "../../Datasets"
    output_base_path = "../../Downstream_Results"

    for dataset, columns in datasets.items():
        print('-' * 70)
        print(f"{dataset}:Processing...")
        target = columns["target_column"]
        clean_path = os.path.join(input_base_path, dataset, 'clean.csv')
        clean_df = pd.read_csv(clean_path).astype(str)
        clean_df.fillna('nan', inplace=True)
        feature_schema = list(clean_df.columns)
        feature_schema.remove(target)

        # 首先在clean数据上找到最优参数
        print("正在搜索最优参数...")
        df_encoded = pd.get_dummies(clean_df[feature_schema])
        train_indices, test_indices = train_test_split(range(len(df_encoded)), test_size=0.2, random_state=0)
        X_train = df_encoded.iloc[train_indices]
        y_train = clean_df[target].iloc[train_indices]
        best_params = get_best_mlp_params(X_train, y_train)
        print(f"找到最优参数: {best_params}")

        # 初始化结果列表
        results = []

        # 处理清洁数据
        res_dict = testing_func(clean_df, clean_df, target, feature_schema, best_params)
        for algm in res_dict:
            clean_for_pg_mse = res_dict[algm][0]
            clean_for_pg_mae = res_dict[algm][1]
            results.append(["clean.csv", res_dict[algm][0], res_dict[algm][1], 0, 0])
            print("'clean.csv' is ok.")
            print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, 0, 0")

        # 处理脏数据
        for rate in Missing_rate:
            input_dirty_file = os.path.join(input_base_path, dataset, "null", f'dirty-{rate}.csv')
            dirty_df = pd.read_csv(input_dirty_file).astype(str)
            dirty_df.fillna('nan', inplace=True)
            res_dict = testing_func(dirty_df, clean_df, target, feature_schema, best_params)
            for algm in res_dict:
                print(f"'dirty-{rate}.csv' is ok.")
                if res_dict[algm][0] < clean_for_pg_mse:
                    dirty_for_pg_mse = 0
                else:
                    dirty_for_pg_mse = (res_dict[algm][0] - clean_for_pg_mse) / clean_for_pg_mse
                if res_dict[algm][1] < clean_for_pg_mae:
                    dirty_for_pg_mae = 0
                else:
                    dirty_for_pg_mae = (res_dict[algm][1] - clean_for_pg_mae) / clean_for_pg_mae
                results.append([f"dirty-{rate}.csv", res_dict[algm][0], res_dict[algm][1], dirty_for_pg_mse, dirty_for_pg_mae])
                print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {dirty_for_pg_mse}, {dirty_for_pg_mae}")

        # 处理填补数据
        for model in Imputation_Algorithms:
            for rate in Missing_rate:
                imputed_path = os.path.join(input_base_path, dataset, "Imputation", f'null-{model}',
                                            f'dirty-{model}-{rate}.csv')
                imputed_df = pd.read_csv(imputed_path).astype(str)
                imputed_df.fillna('nan', inplace=True)
                res_dict = testing_func(imputed_df, clean_df, target, feature_schema, best_params)
                for algm in res_dict:
                    print(f"'dirty-{model}-{rate}.csv' is ok.")
                    if res_dict[algm][0] < clean_for_pg_mse:
                        dirty_for_pg_mse = 0
                    else:
                        dirty_for_pg_mse = (res_dict[algm][0] - clean_for_pg_mse) / clean_for_pg_mse
                    if res_dict[algm][1] < clean_for_pg_mae:
                        dirty_for_pg_mae = 0
                    else:
                        dirty_for_pg_mae = (res_dict[algm][1] - clean_for_pg_mae) / clean_for_pg_mae
                    results.append([f"dirty-{model}-{rate}.csv", res_dict[algm][0], res_dict[algm][1], dirty_for_pg_mse,
                                    dirty_for_pg_mae])
                    print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {dirty_for_pg_mse}, {dirty_for_pg_mae}")

        output_results_file = os.path.join(output_base_path, "regression", dataset,
                                           f"mlp-imputation-results-{dataset}.csv")
        dir_path = os.path.dirname(output_results_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        results_df = pd.DataFrame(results, columns=["File Name", "MSE", "MAE", "PG(MSE)", "PG(MAE)"])
        results_df.to_csv(output_results_file, index=False)