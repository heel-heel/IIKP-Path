import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os

params = {
    "generate_timeseries":{
        "look_back": 10,
        "mlp_param":{
            "solver": 'sgd',
            "max_iter": 1000,
            "learning_rate_init": 0.1,
            "hidden_layer_sizes": (100, 50, 50, 20),
            "early_stopping": True,
            "alpha": 0.01,
            "activation": 'relu'
            }
    }
}

Ingredients = ['resid', 'trend', 'seasonal']
portion_list = [50]
corr_list = [70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]
#corr_list = [70, 98]

def create_dataset(dataset, look_back=12):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

dataset = "generate_timeseries"

print('-' * 70)
print(f"{dataset}: Processing...")
target_column = "y_combined"
results = []

# get the parameters of the dataset
dataset_param = params[dataset]
look_back = dataset_param["look_back"]
mlp_param = dataset_param["mlp_param"]

# ==================== Processing the clean data ====================
input_clean_file = os.path.join("../generate_timeseries_results", "synthetic_time_series_balanced.csv")
clean_data = pd.read_csv(input_clean_file)
target_clean = clean_data[target_column].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler.fit_transform(target_clean)

#look_back = 12
X, y = create_dataset(target_scaled, look_back)
X = X.reshape(X.shape[0], look_back)

split_idx = int(len(X) * 0.7)
X_train_clean, X_test_clean = X[:split_idx], X[split_idx:]
y_train_clean, y_test_clean = y[:split_idx], y[split_idx:]
y_test_scaled = scaler.inverse_transform(y_test_clean.reshape(-1, 1))  # 真实标签

# model training
mlp_model = MLPRegressor(
    **mlp_param,
    random_state=42,
    verbose=False
)
mlp_model.fit(X_train_clean, y_train_clean)
predictions = mlp_model.predict(X_test_clean)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
mae = mean_absolute_error(y_test_scaled, predictions)
clean_for_pg_rmse = rmse
clean_for_pg_mae = mae
results.append(["clean.csv", rmse, mae, 0, 0])
print("clean.csv", rmse, mae, 0, 0)
#printf("clean.csv", rmse, mae, 0, 0)

# ==================== Processing the generated data ====================
for portion in portion_list:
    for corr in corr_list:
        for ingredient in Ingredients:
            input_dirty_file = os.path.join(
                "../decompose_change_good", "decompose_change_good_results",
                f"{ingredient}", f"dirty-{ingredient}-{portion}-{corr}.csv"
            )

            test_data = pd.read_csv(input_dirty_file)
            target_test = test_data[target_column].values.reshape(-1, 1)

            target_scaled_test = scaler.transform(target_test)

            X_dirty, y_dirty = create_dataset(target_scaled_test, look_back)
            X_dirty = X_dirty.reshape(X_dirty.shape[0], look_back)

            assert len(X_dirty) == len(X), "The number of classification_overview data samples does not match the number of clean data samples!"
            X_train_dirty, X_test_dirty = X_dirty[:split_idx], X_dirty[split_idx:]
            y_train_dirty, y_test_dirty = y_dirty[:split_idx], y_dirty[split_idx:]

            # model training
            mlp_model_dirty = MLPRegressor(
                **mlp_param,
                random_state=42,
                verbose=False
            )
            mlp_model_dirty.fit(X_train_dirty, y_train_dirty)
            predictions_dirty = mlp_model_dirty.predict(X_test_dirty)
            predictions_dirty = scaler.inverse_transform(predictions_dirty.reshape(-1, 1))  # Key modification points

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
            results.append([f"dirty-{ingredient}-{portion}-{corr}.csv", rmse, mae, dirty_for_pg_rmse, dirty_for_pg_mae])
            print(f"dirty-{ingredient}-{portion}-{corr}.csv", rmse, mae, dirty_for_pg_rmse, dirty_for_pg_mae)

output_results_path = "results"
os.makedirs(output_results_path, exist_ok=True)
output_results_file = os.path.join(output_results_path, f"mlp-decompose_change_good-results-{dataset}.csv")
results_df = pd.DataFrame(results, columns=["File Name", "RMSE", "MAE", "PG(RMSE)", "PG(MAE)"])
results_df.to_csv(output_results_file, index=False)