import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import random

datasets = {
    #"M4-Daily": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Weekly": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    #"M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"},

    "ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},
    "ETTm1": {"target_column": "OT", "nonnumerical_column": "date"},
    "Illness": {"target_column": "OT", "nonnumerical_column": "date"},
    "Exchange": {"target_column": "OT", "nonnumerical_column": "date"},
    "Weather": {"target_column": "OT", "nonnumerical_column": "date"}
}
params = {
    "ETTh1":{
        "look_back": 20,
        "lightts_param":{
            "num_epochs":1000,
            'e_layers': 2,#no effect
            "d_model":68,
            "dropout": 0.1,#no effect
            "early_stopping":False,
            "learning_rate": 0.0001
            }
    },
    "ETTm1":{
        "look_back": 6,
        "lightts_param":{
            "num_epochs":1000,
            'e_layers': 2,#no effect
            "d_model":60,
            "dropout": 0.1,#no effect
            "early_stopping":True,
            "learning_rate": 0.0001
            }
    },
    "Illness":{
        "look_back": 10,
        "lightts_param":{
            "num_epochs":100,
            'e_layers': 2,#no effect
            "d_model":128,
            "dropout": 0.1,#no effect
            "early_stopping":True,
            "learning_rate": 0.01
            }
    },
    "Exchange":{
        "look_back": 8,
        "lightts_param":{
            "num_epochs":100,
            'e_layers': 2,#no effect
            "d_model":64,
            "dropout": 0.1,#no effect
            "early_stopping":True,
            "learning_rate": 0.01
            }
    },
    "Weather":{
        "look_back": 5,
        "lightts_param":{
            "num_epochs":500,
            'e_layers': 2,#no effect
            "d_model":92,
            "dropout": 0.1,#no effect
            "early_stopping":True,
            "learning_rate": 0.0001
            }
    },




    "M4-Daily":{
        "look_back": 6,
        "lightts_param":{
            "num_epochs":1000,
            'e_layers': 2,#no effect
            "d_model":68,
            "dropout": 0.1,#no effect
            "early_stopping":False,
            "learning_rate": 0.0001
            }
    },
    "M4-Weekly":{
        "look_back": 13,
        "lightts_param":{
            "num_epochs":1000,
            'e_layers': 2,#no effect
            "d_model":64,
            "dropout": 0.1,#no effect
            "early_stopping":False,
            "learning_rate": 0.0001
            }
    },
    "M4-Monthly":{
        "look_back": 13,
        "lightts_param":{
            "num_epochs":270,
            'e_layers': 2,#no effect
            "d_model":72,
            "dropout": 0.1,#no effect
            "early_stopping":False,
            "learning_rate": 0.0001
            }
    },
    "M4-Quarterly":{
        "look_back": 12,
        "lightts_param":{
            "num_epochs":300,
            'e_layers': 2,#no effect
            "d_model":72,
            "dropout": 0.1,#no effect
            "early_stopping":False,
            "learning_rate": 0.0001
            }
    },
    "M4-Yearly":{
        "look_back": 9,
        "lightts_param":{
            "num_epochs":200,
            'e_layers': 2,#no effect
            "d_model":64,
            "dropout": 0.1,#no effect
            "early_stopping":False,
            "learning_rate": 0.0001
        }
    }
}

Ingredients = ['resid', 'trend', 'seasonal']
portion_list = [50]
corr_list = list(range(70, 99, 2))

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)


# LightTS implementation
class IEBlock(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_node):
        super(IEBlock, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_node = num_node

        self._build()

    def _build(self):
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 4)
        )

        self.channel_proj = nn.Linear(self.num_node, self.num_node)
        torch.nn.init.eye_(self.channel_proj.weight)

        self.output_proj = nn.Linear(self.hid_dim // 4, self.output_dim)

    def forward(self, x):
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class LightTSModel(nn.Module):
    def __init__(self, configs, chunk_size=24):
        super(LightTSModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.chunk_size = min(configs.pred_len, configs.seq_len, chunk_size)

        # Adjust sequence length to be divisible by chunk size
        if self.seq_len % self.chunk_size != 0:
            self.seq_len += (self.chunk_size - self.seq_len % self.chunk_size)
        self.num_chunks = self.seq_len // self.chunk_size

        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.dropout = configs.dropout
        self._build()

    def _build(self):
        self.layer_1 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4,
            num_node=self.num_chunks
        )

        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)

        self.layer_2 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4,
            num_node=self.num_chunks
        )

        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)

        self.layer_3 = IEBlock(
            input_dim=self.d_model // 2,
            hid_dim=self.d_model // 2,
            output_dim=self.pred_len,
            num_node=self.enc_in
        )

        self.ar = nn.Linear(self.seq_len, self.pred_len)

    def encoder(self, x):
        B, T, N = x.size()

        highway = self.ar(x.permute(0, 2, 1))
        highway = highway.permute(0, 2, 1)

        # continuous sampling
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N)
        x1 = x1.permute(0, 3, 2, 1)
        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)

        # interval sampling
        x2 = x.reshape(B, self.chunk_size, self.num_chunks, N)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)

        x3 = torch.cat([x1, x2], dim=-1)
        x3 = x3.reshape(B, N, -1)
        x3 = x3.permute(0, 2, 1)

        out = self.layer_3(x3)
        out = out + highway
        return out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.encoder(x_enc)

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


def train_evaluate_lightts_early_stopping_false(X_train, y_train, X_test, y_test, scaler, best_look_back, config_params):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = Configs()
    configs.seq_len = best_look_back
    configs.e_layers = config_params['e_layers']
    configs.d_model = config_params['d_model']
    configs.dropout = config_params['dropout']
    configs.learning_rate = config_params['learning_rate']

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1, 1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    model = LightTSModel(configs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)
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

# early stopping
def train_evaluate_lightts_early_stopping_true(X_train, y_train, X_test, y_test, scaler, best_look_back, config_params):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = Configs()
    configs.seq_len = best_look_back
    configs.e_layers = config_params['e_layers']
    configs.d_model = config_params['d_model']
    configs.dropout = config_params['dropout']
    configs.learning_rate = config_params['learning_rate']

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1, 1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    model = LightTSModel(configs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)
    criterion = nn.MSELoss()

    num_epochs = config_params['num_epochs']
    early_stopping_patience = 10
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

        # evaluate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor, None, None, None)
            val_loss = criterion(val_outputs, torch.FloatTensor(y_test).view(-1, 1, 1).to(device)).item()

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
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
    lightts_param = dataset_param["lightts_param"]

    # ==================== Processing the clean data  ====================
    clean_path = os.path.join(base_path, dataset, "clean.csv")
    clean_data = pd.read_csv(clean_path)
    target = clean_data[target_column].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    target_scaled = scaler.fit_transform(target)

    X, y = create_dataset(target_scaled, look_back)

    # using the best parameters
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if lightts_param['early_stopping'] == True:
        clean_rmse, clean_mae = train_evaluate_lightts_early_stopping_true(X_train, y_train, X_test, y_test, scaler,
                                                                           look_back, lightts_param)
    elif lightts_param['early_stopping'] == False:
        clean_rmse, clean_mae = train_evaluate_lightts_early_stopping_false(X_train, y_train, X_test, y_test, scaler,
                                                                            look_back, lightts_param)

    results.append(["clean.csv", clean_rmse, clean_mae, 0, 0])
    print("'clean.csv' is ok.")
    print(f"{clean_rmse}, {clean_mae}, 0, 0")

    # ==================== Process the generated data ======================
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

                if lightts_param['early_stopping'] == True:
                    rmse, mae = train_evaluate_lightts_early_stopping_true(X_train_d, y_train_d, X_test_d, y_test,
                                                                           scaler, look_back, lightts_param)
                elif lightts_param['early_stopping'] == False:
                    rmse, mae = train_evaluate_lightts_early_stopping_false(X_train_d, y_train_d, X_test_d, y_test,
                                                                            scaler, look_back, lightts_param)

                pg_rmse = (rmse - clean_rmse) / clean_rmse if rmse > clean_rmse else 0
                pg_mae = (mae - clean_mae) / clean_mae if mae > clean_mae else 0
                results.append([f"dirty-{ingredient}-{portion}-{corr}.csv", rmse, mae, pg_rmse, pg_mae])
                print(f"'dirty-{ingredient}-{portion}-{corr}.csv' is ok.")
                print(f"{rmse}, {mae}, {pg_rmse}, {pg_mae}")

    output_dir = os.path.join("../../Downstream_Results", "timeseries", dataset)
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results, columns=["File Name", "RMSE", "MAE", "PG(RMSE)", "PG(MAE)"])
    results_df.to_csv(os.path.join(output_dir, f"lightts-decompose_change_good-results-{dataset}.csv"), index=False)

print("All tasks completed!")