import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import os
import random

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
        "tsmixer_param":{
            "num_epochs":70,
            "e_layers":3,
            "d_model":15,
            "dropout":0.15,
            "early_stopping":False
            }
    },
    "M4-Weekly":{
        "look_back": 13,
        "tsmixer_param":{
            "num_epochs":30,
            "e_layers":3,
            "d_model":25,
            "dropout":0.3,
            "early_stopping":False
            }
    },
    "M4-Monthly":{
        "look_back": 13,
        "tsmixer_param":{
            "num_epochs":200,
            "e_layers":3,
            "d_model":15,
            "dropout":0.15,
            "early_stopping":False
            }
    },
    "M4-Quarterly":{
        "look_back": 12,
        "tsmixer_param":{
            "num_epochs":80,
            "e_layers":3,
            "d_model":20,
            "dropout":0.15,
            "early_stopping":False
            }
    },
    "M4-Yearly":{
        "look_back": 9,
        "tsmixer_param":{
            "num_epochs":100,
            "e_layers":3,
            "d_model":15,
            "dropout":0.25,
            "early_stopping":False
        }
    }
}

Ingredients = ['resid', 'trend', 'seasonal']
portion_list = [50]
corr_list = list(range(70, 99, 2))


# 设置全局随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)


# TSLib中的对应实现
class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )
        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs) for _ in range(configs.e_layers)])
        self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]


class Configs:
    def __init__(self):
        self.e_layers = 2
        self.d_model = 64
        self.enc_in = 1
        self.dropout = 0.1
        self.pred_len = 1
        self.seq_len = None


def create_dataset(dataset, look_back=12):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    X = np.array(X).reshape(-1, look_back, 1)
    Y = np.array(Y)
    return X, Y


def train_evaluate_tsmixer_early_stopping_false(X_train, y_train, X_test, y_test, scaler, look_back, config_params):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = Configs()
    configs.seq_len = look_back
    configs.e_layers = config_params['e_layers']
    configs.d_model = config_params['d_model']
    configs.dropout = config_params['dropout']

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1, 1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    model = Model(configs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = config_params['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X, None, None, None)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor, None, None, None)
    predictions = outputs.cpu().squeeze().numpy()
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
    mae = mean_absolute_error(y_test_scaled, predictions)
    return rmse, mae


# 早停版本
def train_evaluate_tsmixer_early_stopping_true(X_train, y_train, X_test, y_test, scaler, look_back, config_params):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = Configs()
    configs.seq_len = look_back
    configs.e_layers = config_params['e_layers']
    configs.d_model = config_params['d_model']
    configs.dropout = config_params['dropout']

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1, 1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    model = Model(configs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = config_params['num_epochs']
    early_stopping_patience = 10  # 设置早停的耐心轮次
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X, None, None, None)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor, None, None, None)
            val_loss = criterion(val_outputs, torch.FloatTensor(y_test).view(-1, 1, 1).to(device)).item()

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                # print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor, None, None, None)
    predictions = outputs.cpu().squeeze().numpy()
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
    mae = mean_absolute_error(y_test_scaled, predictions)
    return rmse, mae


for dataset, columns in datasets.items():
    print('-' * 70)
    print(f"{dataset}: Processing...")
    target_column = columns["target_column"]
    base_path = "../../Datasets"
    results = []

    # 获取当前数据集参数
    dataset_param = params[dataset]
    look_back = dataset_param["look_back"]
    tsmixer_param = dataset_param["tsmixer_param"]

    # ==================== Clean数据处理 ====================
    clean_path = os.path.join(base_path, dataset, "clean.csv")
    clean_data = pd.read_csv(clean_path)
    target = clean_data[target_column].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    target_scaled = scaler.fit_transform(target)

    X, y = create_dataset(target_scaled, look_back)

    # 使用最佳参数在整个训练集上训练
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if tsmixer_param['early_stopping'] == True:
        clean_rmse, clean_mae = train_evaluate_tsmixer_early_stopping_true(X_train, y_train, X_test, y_test, scaler, look_back, tsmixer_param)
    elif tsmixer_param['early_stopping'] == False:
        clean_rmse, clean_mae = train_evaluate_tsmixer_early_stopping_false(X_train, y_train, X_test, y_test, scaler, look_back, tsmixer_param)

    results.append(["clean.csv", clean_rmse, clean_mae, 0, 0])
    print("'clean.csv' is ok.")
    print(f"{clean_rmse}, {clean_mae}, 0, 0")

    # ==================== 处理生成数据 ==================
    for portion in portion_list:
        for corr in corr_list:
            for ingredient in Ingredients:
                input_dirty_file = os.path.join(
                    base_path, dataset, "Mechanism", "timeseries", "decompose_change_good",
                    f"{ingredient}", f"dirty-{ingredient}-{portion}-{corr}.csv"
                )
                test_data = pd.read_csv(input_dirty_file)
                target_test = test_data[target_column].values.reshape(-1, 1)
                target_scaled_test = scaler.transform(target_test)

                X_dirty, y_dirty = create_dataset(target_scaled_test, look_back)
                X_train_d, X_test_d = X_dirty[:split_idx], X_dirty[split_idx:]
                y_train_d, y_test_d = y_dirty[:split_idx], y_dirty[split_idx:]

                if tsmixer_param['early_stopping'] == True:
                    rmse, mae = train_evaluate_tsmixer_early_stopping_true(X_train_d, y_train_d, X_test_d, y_test, scaler, look_back, tsmixer_param)
                elif tsmixer_param['early_stopping'] == False:
                    rmse, mae = train_evaluate_tsmixer_early_stopping_false(X_train_d, y_train_d, X_test_d, y_test, scaler, look_back, tsmixer_param)

                pg_rmse = (rmse - clean_rmse) / clean_rmse if rmse > clean_rmse else 0
                pg_mae = (mae - clean_mae) / clean_mae if mae > clean_mae else 0
                results.append([f"dirty-{ingredient}-{portion}-{corr}.csv", rmse, mae, pg_rmse, pg_mae])
                print(f"'dirty-{ingredient}-{portion}-{corr}.csv' is ok.")
                print(f"{rmse}, {mae}, {pg_rmse}, {pg_mae}")

    # 保存结果
    output_dir = os.path.join("../../Downstream_Results", "timeseries", dataset)
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results, columns=["File Name", "RMSE", "MAE", "PG(RMSE)", "PG(MAE)"])
    results_df.to_csv(os.path.join(output_dir, f"tsmixer-decompose_change_good-results-{dataset}.csv"), index=False)

print("All tasks completed!")