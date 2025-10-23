import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os

datasets = {
    #"M4-Hourly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Daily": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Weekly": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"}
}
params = {
    "M4-Daily":{
        "look_back": 6,
        "mlp_param":{
            "solver": 'sgd',
            "max_iter": 3000,
            "learning_rate_init": 0.1,
            "hidden_layer_sizes": (75,),
            "early_stopping": True,
            "alpha": 0.01,
            "activation": 'tanh'
            }
    },
    "M4-Weekly":{
        "look_back": 13,
        "mlp_param":{
            "solver": 'sgd',
            "max_iter": 1000,
            "learning_rate_init": 0.1,
            "hidden_layer_sizes": (50, 20),
            "early_stopping": True,
            "alpha": 0.1,
            "activation": 'tanh'
            }
    },
    "M4-Monthly":{
        "look_back": 13,
        "mlp_param":{
            "solver": 'sgd',
            "max_iter": 1000,
            "learning_rate_init": 0.1,
            "hidden_layer_sizes": (80, 30),
            "early_stopping": True,
            "alpha": 0.01,
            "activation": 'tanh'
            }
    },
    "M4-Quarterly":{
        "look_back": 12,
        "mlp_param":{
            "solver": 'adam',
            "max_iter": 3000,
            "learning_rate_init": 0.01,
            "hidden_layer_sizes": (50, 20, 10),
            "early_stopping": True,
            "alpha": 0.001,
            "activation": 'tanh'
            }
    },
    "M4-Yearly":{
        "look_back": 9,
        "mlp_param":{
            "solver": 'sgd',
            "max_iter": 3000,
            "learning_rate_init": 0.001,
            "hidden_layer_sizes": (60, 20),
            "early_stopping": True,
            "alpha": 0.01,
            "activation": 'relu'
        }
    }
}

Ingredients = ['resid', 'trend', 'seasonal']
portion_list = [50]
corr_list = list(range(70, 99, 2))

def create_dataset(dataset, look_back=12):
    """将时间序列转换为监督学习格式"""
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

for dataset, columns in datasets.items():
    print('-' * 70)
    print(f"{dataset}: Processing...")
    target_column = columns["target_column"]
    base_path = "../../Datasets"
    results = []

    # 获取当前数据集参数
    dataset_param = params[dataset]
    look_back = dataset_param["look_back"]
    mlp_param = dataset_param["mlp_param"]

    # ==================== 处理干净数据 ====================
    input_clean_file = os.path.join(base_path, dataset, "clean.csv")
    clean_data = pd.read_csv(input_clean_file)
    target_clean = clean_data[target_column].values.reshape(-1, 1)

    # 数据标准化 (仅使用干净数据训练scaler)
    scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaled = scaler.fit_transform(target_clean)

    # 转换为监督学习问题
    #look_back = 12
    X, y = create_dataset(target_scaled, look_back)
    X = X.reshape(X.shape[0], look_back)

    # 按时间顺序划分测试集 (最后30%)
    split_idx = int(len(X) * 0.7)
    X_train_clean, X_test_clean = X[:split_idx], X[split_idx:]
    y_train_clean, y_test_clean = y[:split_idx], y[split_idx:]
    y_test_scaled = scaler.inverse_transform(y_test_clean.reshape(-1, 1))  # 真实标签

    # 训练模型
    mlp_model = MLPRegressor(
        **mlp_param,
        random_state=42,
        verbose=1
    )
    mlp_model.fit(X_train_clean, y_train_clean)
    predictions = mlp_model.predict(X_test_clean)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
    mae = mean_absolute_error(y_test_scaled, predictions)
    clean_for_pg_rmse = rmse
    clean_for_pg_mae = mae
    results.append(["clean.csv", rmse, mae, 0, 0])

    # ==================== 处理生成数据 ====================
    for portion in portion_list:
        for corr in corr_list:
            for ingredient in Ingredients:
                input_dirty_file = os.path.join(
                    base_path, dataset, "Mechanism", "timeseries", "decompose_change_good",
                    f"{ingredient}", f"dirty-{ingredient}-{portion}-{corr}.csv"
                )

                test_data = pd.read_csv(input_dirty_file)
                target_test = test_data[target_column].values.reshape(-1, 1)

                # 使用干净数据的scaler标准化
                target_scaled_test = scaler.transform(target_test)

                # 转换为监督学习问题
                X_dirty, y_dirty = create_dataset(target_scaled_test, look_back)
                X_dirty = X_dirty.reshape(X_dirty.shape[0], look_back)

                # 确保测试集索引对齐
                assert len(X_dirty) == len(X), "测试数据样本数与干净数据不一致!"
                X_train_dirty, X_test_dirty = X_dirty[:split_idx], X_dirty[split_idx:]
                y_train_dirty, y_test_dirty = y_dirty[:split_idx], y_dirty[split_idx:]

                # 训练模型
                mlp_model_dirty = MLPRegressor(
                    **mlp_param,
                    random_state=42,
                    verbose=1
                )
                mlp_model_dirty.fit(X_train_dirty, y_train_dirty)
                predictions_dirty = mlp_model_dirty.predict(X_test_dirty)
                predictions_dirty = scaler.inverse_transform(predictions_dirty.reshape(-1, 1))  # 关键修改点

                # 计算指标
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

    # 保存结果
    output_base_path = "../../Downstream_Results"
    output_results_path = os.path.join(output_base_path, "timeseries", dataset)
    os.makedirs(output_results_path, exist_ok=True)
    output_results_file = os.path.join(output_results_path, f"mlp-decompose_change_good-results-{dataset}.csv")
    results_df = pd.DataFrame(results, columns=["File Name", "RMSE", "MAE", "PG(RMSE)", "PG(MAE)"])
    results_df.to_csv(output_results_file, index=False)