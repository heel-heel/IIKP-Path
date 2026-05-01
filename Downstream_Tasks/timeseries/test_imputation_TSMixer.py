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


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)


# TSLib
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


def train_evaluate_tsmixer_early_stopping_false(X_train, y_train, X_test, y_test, scaler, best_look_back,
                                                config_params):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = Configs()
    configs.seq_len = best_look_back
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


# early stopping
def train_evaluate_tsmixer_early_stopping_true(X_train, y_train, X_test, y_test, scaler, best_look_back, config_params):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = Configs()
    configs.seq_len = best_look_back
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

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        # early stopping
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

datasets = {
    "ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},
    "ETTm1": {"target_column": "OT", "nonnumerical_column": "date"},
    "Illness": {"target_column": "OT", "nonnumerical_column": "date"},
    "Exchange": {"target_column": "OT", "nonnumerical_column": "date"},
    "Weather": {"target_column": "OT", "nonnumerical_column": "date"}
}

look_back_settings = {
    "M4-Daily": 6,
    "M4-Weekly": 13,
    "M4-Monthly": 13,
    "M4-Quarterly": 12,
    "M4-Yearly": 9,
    "ETTh1": 20,
    "ETTm1": 6,
    "Illness": 10,
    "Exchange": 8,
    "Weather": 5,
}

Imputation_Algorithms = ['mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
Mechanism = ["MCAR", "MAR", "MNAR"]

# Define the parameter grid for random search
param_dist = {
    'num_epochs': [20, 30, 50, 60, 70, 80, 100, 120, 150, 200, 300, 400, 500, 1000],
    'e_layers': [1, 2, 3, 4],
    'd_model': [10, 15, 20, 25, 50, 60, 70, 80, 100],
    'dropout': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
    'early_stopping': [True, False]
}

tscv = TimeSeriesSplit(n_splits=2)

for dataset, columns in datasets.items():
    print(f'Processing {dataset}...')
    target_col = columns["target_column"]
    base_path = "../../Datasets"


    for pattern in Mechanism:
        results = []
        best_look_back = look_back_settings[dataset]

        # ==================== Process the clean data ====================
        clean_path = os.path.join(base_path, dataset, "clean.csv")
        clean_data = pd.read_csv(clean_path)
        target = clean_data[target_col].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        target_scaled = scaler.fit_transform(target)

        X, y = create_dataset(target_scaled, best_look_back)

        tscv_scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #n_iter = 50
            n_iter = 1
            param_combinations = []
            for _ in range(n_iter):
                params = {
                    'num_epochs': random.choice(param_dist['num_epochs']),
                    'e_layers': random.choice(param_dist['e_layers']),
                    'd_model': random.choice(param_dist['d_model']),
                    'dropout': random.choice(param_dist['dropout']),
                    'early_stopping': random.choice(param_dist['early_stopping'])
                }
                param_combinations.append(params)

            best_rmse = float('inf')
            best_mae = float('inf')
            best_params = None

            for params in param_combinations:
                if params['early_stopping'] == True:
                    current_rmse, current_mae = train_evaluate_tsmixer_early_stopping_true(X_train, y_train, X_test, y_test,
                                                                                           scaler, best_look_back, params)
                elif params['early_stopping'] == False:
                    current_rmse, current_mae = train_evaluate_tsmixer_early_stopping_false(X_train, y_train, X_test,
                                                                                            y_test, scaler, best_look_back,
                                                                                            params)
                print(f"Testing params: {params}, RMSE: {current_rmse}, MAE: {current_mae}")

                if current_rmse < best_rmse:
                    best_rmse = current_rmse
                    best_mae = current_mae
                    best_params = params

            tscv_scores.append((best_rmse, best_mae, best_params))

        # the best parameters
        best_cv_params = min(tscv_scores, key=lambda x: x[0])[2]
        print(f"Best parameters from CV: {best_cv_params}")

        # using best parameters
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if best_cv_params['early_stopping'] == True:
            clean_rmse, clean_mae = train_evaluate_tsmixer_early_stopping_true(X_train, y_train, X_test, y_test, scaler,
                                                                               best_look_back, best_cv_params)
        elif best_cv_params['early_stopping'] == False:
            clean_rmse, clean_mae = train_evaluate_tsmixer_early_stopping_false(X_train, y_train, X_test, y_test, scaler,
                                                                                best_look_back, best_cv_params)

        results.append(["clean.csv", clean_rmse, clean_mae, 0, 0])
        print("'clean.csv' is ok.")
        print(f"{clean_rmse}, {clean_mae}, 0, 0")

        # ==================== Process the dirty data ======================

        for rate in Missing_rate:
            dirty_path = os.path.join(base_path, dataset, "null", pattern, f"dirty-{rate}.csv")
            dirty_data = pd.read_csv(dirty_path).fillna(0)
            target_dirty = dirty_data[target_col].values.reshape(-1, 1)
            target_scaled_dirty = scaler.transform(target_dirty)
            X_dirty, y_dirty = create_dataset(target_scaled_dirty, best_look_back)
            X_train_d, X_test_d = X_dirty[:split_idx], X_dirty[split_idx:]
            y_train_d, y_test_d = y_dirty[:split_idx], y_dirty[split_idx:]

            if best_cv_params['early_stopping'] == True:
                rmse, mae = train_evaluate_tsmixer_early_stopping_true(X_train_d, y_train_d, X_test_d, y_test, scaler,
                                                                       best_look_back, best_cv_params)
            elif best_cv_params['early_stopping'] == False:
                rmse, mae = train_evaluate_tsmixer_early_stopping_false(X_train_d, y_train_d, X_test_d, y_test, scaler,
                                                                        best_look_back, best_cv_params)

            pg_rmse = (rmse - clean_rmse) / clean_rmse if rmse > clean_rmse else 0
            pg_mae = (mae - clean_mae) / clean_mae if mae > clean_mae else 0
            results.append([f"dirty-{rate}.csv", rmse, mae, pg_rmse, pg_mae])
            print(f"'dirty-{rate}.csv' is ok.")
            print(f"{rmse}, {mae}, {pg_rmse}, {pg_mae}")

        # ==================== Process the imputed data ==================
        for model in Imputation_Algorithms:
            for rate in Missing_rate:
                imputed_path = os.path.join(base_path, dataset, "Imputation", pattern, f"null-{model}", f"dirty-{model}-{rate}.csv")
                imputed_data = pd.read_csv(imputed_path)
                target_imp = imputed_data[target_col].values.reshape(-1, 1)
                target_scaled_imp = scaler.transform(target_imp)
                X_imp, y_imp = create_dataset(target_scaled_imp, best_look_back)
                X_train_i, X_test_i = X_imp[:split_idx], X_imp[split_idx:]
                y_train_i, y_test_i = y_imp[:split_idx], y_imp[split_idx:]

                if best_cv_params['early_stopping'] == True:
                    rmse, mae = train_evaluate_tsmixer_early_stopping_true(X_train_i, y_train_i, X_test_i, y_test, scaler,
                                                                           best_look_back, best_cv_params)
                elif best_cv_params['early_stopping'] == False:
                    rmse, mae = train_evaluate_tsmixer_early_stopping_false(X_train_i, y_train_i, X_test_i, y_test, scaler,
                                                                            best_look_back, best_cv_params)

                pg_rmse = (rmse - clean_rmse) / clean_rmse if rmse > clean_rmse else 0
                pg_mae = (mae - clean_mae) / clean_mae if mae > clean_mae else 0
                results.append([f"dirty-{model}-{rate}.csv", rmse, mae, pg_rmse, pg_mae])
                print(f"'dirty-{model}-{rate}.csv' is ok.")
                print(f"{rmse}, {mae}, {pg_rmse}, {pg_mae}")

        output_dir = os.path.join("../../Downstream_Results", "timeseries", dataset, pattern)
        os.makedirs(output_dir, exist_ok=True)
        results_df = pd.DataFrame(results, columns=["File Name", "RMSE", "MAE", "PG(RMSE)", "PG(MAE)"])
        results_df.to_csv(os.path.join(output_dir, f"tsmixer-imputation-results-{dataset}.csv"), index=False)

print("All tasks completed!")