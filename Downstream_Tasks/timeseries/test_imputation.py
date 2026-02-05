#def warn(*args, **kwargs):
#    pass
#import warnings
#warnings.warn = warn

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import os

datasets = {
    "ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},
    "ETTm1": {"target_column": "OT", "nonnumerical_column": "date"},
    "Illness": {"target_column": "OT", "nonnumerical_column": "date"},
    "Exchange": {"target_column": "OT", "nonnumerical_column": "date"},
    "Weather": {"target_column": "OT", "nonnumerical_column": "date"}
}
Imputation_Algorithms = ['mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


def create_dataset(dataset, look_back=12):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Define the parameter grid for random search
param_grid = {
    'hidden_layer_sizes': [
        (10,), (20,), (50,), (60,), (70,), (75,), (80,), (100,), (120, ),
        (10, 5), (20, 10), (40, 10), (45, 20), (50, 10), (50, 15), (50, 20), (50, 25), (50, 30), (55, 20), (55, 25),
        (60, 10), (60, 15), (60, 20), (60, 25), (65, 20), (65, 25), (70, 25), (70, 30), (80, 30), (80, 50), (100, 50),
        (5, 5, 5), (10, 5, 5), (10, 10, 5), (10, 10, 25), (20, 10, 5), (50, 10, 10), (50, 20, 10), (60, 20, 10),
        (60, 20, 15), (60, 20, 30), (60, 30, 10), (100, 50, 30),
        (20, 10, 10, 5), (25, 25, 10, 5)
    ],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001],
    'learning_rate_init': [0.1, 0.01, 0.001, 0.0001, 0.00001],
    'max_iter': [1000, 2000, 3000, 5000, 8000],
    'early_stopping': [True, False]
}

# Define candidate values for look_back
look_back_candidates = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 24]

for dataset, columns in datasets.items():
    print('-' * 70)
    print(f"{dataset}: Processing...")
    target_column = columns["target_column"]
    base_path = "../../Datasets"
    results = []

    # ==================== Process the clean data ====================
    input_clean_file = os.path.join(base_path, dataset, "clean.csv")
    clean_data = pd.read_csv(input_clean_file)
    target_clean = clean_data[target_column].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaled = scaler.fit_transform(target_clean)

    # Get the best look_back
    print("Finding best look_back value...")
    best_look_back = None
    best_rmse = float('inf')

    for look_back in look_back_candidates:
        X, y = create_dataset(target_scaled, look_back)
        X = X.reshape(X.shape[0], look_back)

        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # valuate look_back using default parameters
        mlp = MLPRegressor(random_state=42)
        mlp.fit(X_train, y_train)
        predictions = mlp.predict(X_test)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))

        if rmse < best_rmse:
            best_rmse = rmse
            best_look_back = look_back

    print(f"Best look_back found: {best_look_back} with RMSE: {best_rmse:.4f}")

    # Get the best look_back to rebuild datasets
    X, y = create_dataset(target_scaled, best_look_back)
    X = X.reshape(X.shape[0], best_look_back)

    split_idx = int(len(X) * 0.7)
    X_train_clean, X_test_clean = X[:split_idx], X[split_idx:]
    y_train_clean, y_test_clean = y[:split_idx], y[split_idx:]
    y_test_scaled = scaler.inverse_transform(y_test_clean.reshape(-1, 1))  # 真实标签

    # Use grid search to find the optimal parameters
    print("Performing grid search on clean data...")
    mlp = MLPRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    # grid_search = GridSearchCV(mlp, param_grid, cv=tscv, scoring='neg_mean_squared_error',
    #                          n_jobs=-1, verbose=1)
    grid_search = RandomizedSearchCV(mlp, param_grid, cv=tscv, scoring='neg_mean_squared_error',
                                     n_jobs=-1, verbose=1, n_iter=50)
    grid_search.fit(X_train_clean, y_train_clean)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    # using best parameters
    mlp_model = MLPRegressor(**best_params, random_state=42)
    mlp_model.fit(X_train_clean, y_train_clean)
    predictions = mlp_model.predict(X_test_clean)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
    mae = mean_absolute_error(y_test_scaled, predictions)
    clean_for_pg_rmse = rmse
    clean_for_pg_mae = mae
    results.append(["clean.csv", rmse, mae, 0, 0])

    # ==================== Process the dirty data ====================
    for rate in Missing_rate:
        input_dirty_file = os.path.join(base_path, dataset, "null", f"dirty-{rate}.csv")
        dirty_data = pd.read_csv(input_dirty_file)
        target_dirty = dirty_data[target_column].fillna(0).values.reshape(-1, 1)

        target_scaled_dirty = scaler.transform(target_dirty)

        X_dirty, y_dirty = create_dataset(target_scaled_dirty, best_look_back)
        X_dirty = X_dirty.reshape(X_dirty.shape[0], best_look_back)

        assert len(X_dirty) == len(X), "The number of classification_overview data samples does not match the number of clean data samples!"
        X_train_dirty, X_test_dirty = X_dirty[:split_idx], X_dirty[split_idx:]
        y_train_dirty, y_test_dirty = y_dirty[:split_idx], y_dirty[split_idx:]

        mlp_model_dirty = MLPRegressor(**best_params, random_state=42)
        mlp_model_dirty.fit(X_train_dirty, y_train_dirty)
        predictions_dirty = mlp_model_dirty.predict(X_test_dirty)
        predictions_dirty = scaler.inverse_transform(predictions_dirty.reshape(-1, 1))

        # calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions_dirty))
        mae = mean_absolute_error(y_test_scaled, predictions_dirty)
        if rmse < clean_for_pg_rmse:
            dirty_for_pg_rmse = 0
        else:
            dirty_for_pg_rmse = (rmse - clean_for_pg_rmse) / clean_for_pg_rmse
        if mae < clean_for_pg_mae:
            dirty_for_pg_mae = 0
        else:
            dirty_for_pg_mae = (mae - clean_for_pg_mae) / clean_for_pg_mae
        results.append([f"dirty-{rate}.csv", rmse, mae, dirty_for_pg_rmse, dirty_for_pg_mae])

    # ==================== Process the imputed data ====================
    for model in Imputation_Algorithms:
        for rate in Missing_rate:
            input_imputed_file = os.path.join(
                base_path, dataset, "Imputation", f"null-{model}",
                f"dirty-{model}-{rate}.csv"
            )
            imputed_data = pd.read_csv(input_imputed_file)
            target_imputed = imputed_data[target_column].values.reshape(-1, 1)

            target_scaled_imputed = scaler.transform(target_imputed)

            X_imputed, y_imputed = create_dataset(target_scaled_imputed, best_look_back)
            X_imputed = X_imputed.reshape(X_imputed.shape[0], best_look_back)

            assert len(X_imputed) == len(X), "The number of classification_overview data samples does not match the number of clean data samples!"
            X_train_imputed, X_test_imputed = X_imputed[:split_idx], X_imputed[split_idx:]
            y_train_imputed, y_test_imputed = y_imputed[:split_idx], y_imputed[split_idx:]

            mlp_model_imputed = MLPRegressor(**best_params, random_state=42)
            mlp_model_imputed.fit(X_train_imputed, y_train_imputed)
            predictions_imputed = mlp_model_imputed.predict(X_test_imputed)
            predictions_imputed = scaler.inverse_transform(predictions_imputed.reshape(-1, 1))

            # calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions_imputed))
            mae = mean_absolute_error(y_test_scaled, predictions_imputed)
            if rmse < clean_for_pg_rmse:
                dirty_for_pg_rmse = 0
            else:
                dirty_for_pg_rmse = (rmse - clean_for_pg_rmse) / clean_for_pg_rmse
            if mae < clean_for_pg_mae:
                dirty_for_pg_mae = 0
            else:
                dirty_for_pg_mae = (mae - clean_for_pg_mae) / clean_for_pg_mae
            results.append([f"dirty-{model}-{rate}.csv", rmse, mae, dirty_for_pg_rmse, dirty_for_pg_mae])

    output_base_path = "../../Downstream_Results"
    output_results_path = os.path.join(output_base_path, "timeseries", dataset)
    os.makedirs(output_results_path, exist_ok=True)
    output_results_file = os.path.join(output_results_path, f"mlp-imputation-results-{dataset}.csv")
    results_df = pd.DataFrame(results, columns=["File Name", "RMSE", "MAE", "PG(RMSE)", "PG(MAE)"])
    results_df.to_csv(output_results_file, index=False)