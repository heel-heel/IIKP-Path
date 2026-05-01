# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import re
import random
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

datasets = {
    "ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},
    "ETTm1": {"target_column": "OT", "nonnumerical_column": "date"},
    "Illness": {"target_column": "OT", "nonnumerical_column": "date"},
    "Exchange": {"target_column": "OT", "nonnumerical_column": "date"},
    "Weather": {"target_column": "OT", "nonnumerical_column": "date"}
}
params = {
    "ETTh1":{
        "scinet_param":{
            "d_layers":1,
            "num_epochs":1,
            "batch_size":32,
            "lr":0.001,
            "current_level":2,
            "kernel_size":2,
            }
    },
    "ETTm1":{
        "scinet_param":{
            "d_layers":1,
            "num_epochs":1,
            "batch_size":64,
            "lr":0.001,
            "current_level":1,
            "kernel_size":2,
            }
    },
    "Illness":{
        "scinet_param":{
            "d_layers":1,
            "num_epochs":1,
            "batch_size":32,
            "lr":0.01,
            "current_level":1,
            "kernel_size":5,
            }
    },
    "Exchange":{
        "scinet_param":{
            "d_layers":1,
            "num_epochs":3,
            "batch_size":128,
            "lr":0.001,
            "current_level":1,
            "kernel_size":5,
            }
    },
    "Weather":{
        "scinet_param":{
            "d_layers":1,
            "num_epochs":1,
            "batch_size":16,
            "lr":0.001,
            "current_level":1,
            "kernel_size":3,
            }
    },
}

Imputation_Algorithms = ['mean', 'median', 'mode', 'si', 'mfi', 'gain', 'midae']
Ingredients = ['resid', 'trend', 'seasonal']
portion_list = [50]
Missing_rate = ['50', '70', '90']


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")
set_seed(42)

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        return self.even(x), self.odd(x)


class CausalConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dropout=0.0):
        super(CausalConvBlock, self).__init__()
        module_list = [
            nn.ReplicationPad1d((kernel_size - 1, kernel_size - 1)),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.Tanh()
        ]
        self.causal_conv = nn.Sequential(*module_list)

    def forward(self, x):
        return self.causal_conv(x)


class SCIBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dropout=0.0):
        super(SCIBlock, self).__init__()
        self.splitting = Splitting()
        self.modules_even = CausalConvBlock(d_model, kernel_size, dropout)
        self.modules_odd = CausalConvBlock(d_model, kernel_size, dropout)
        self.interactor_even = CausalConvBlock(d_model, kernel_size, dropout)
        self.interactor_odd = CausalConvBlock(d_model, kernel_size, dropout)

    def forward(self, x):
        x_even, x_odd = self.splitting(x)
        x_even = x_even.permute(0, 2, 1)
        x_odd = x_odd.permute(0, 2, 1)

        x_even_temp = x_even.mul(torch.exp(self.modules_even(x_odd)))
        x_odd_temp = x_odd.mul(torch.exp(self.modules_odd(x_even)))

        x_even_update = x_even_temp + self.interactor_even(x_odd_temp)
        x_odd_update = x_odd_temp - self.interactor_odd(x_even_temp)

        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)


class SCINet(nn.Module):
    def __init__(self, d_model, current_level=3, kernel_size=5, dropout=0.0):
        super(SCINet, self).__init__()
        self.current_level = current_level
        self.working_block = SCIBlock(d_model, kernel_size, dropout)

        if current_level != 0:
            self.SCINet_Tree_odd = SCINet(d_model, current_level - 1, kernel_size, dropout)
            self.SCINet_Tree_even = SCINet(d_model, current_level - 1, kernel_size, dropout)

    def forward(self, x):
        odd_flag = False
        if x.shape[1] % 2 == 1:
            odd_flag = True
            x = torch.cat((x, x[:, -1:, :]), dim=1)
        x_even_update, x_odd_update = self.working_block(x)
        if odd_flag:
            x_odd_update = x_odd_update[:, :-1]

        if self.current_level == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(
                self.SCINet_Tree_even(x_even_update),
                self.SCINet_Tree_odd(x_odd_update)
            )

    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        min_len = min(even_len, odd_len)

        zipped_data = []
        for i in range(min_len):
            zipped_data.append(even[i].unsqueeze(0))
            zipped_data.append(odd[i].unsqueeze(0))
        if even_len > odd_len:
            zipped_data.append(even[-1].unsqueeze(0))
        return torch.cat(zipped_data, 0).permute(1, 0, 2)


class SCINetModel(nn.Module):
    def __init__(self, configs):
        super(SCINetModel, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Set the number of SCINet stacks
        self.num_stacks = configs.d_layers
        if self.num_stacks == 1:
            self.sci_net_1 = SCINet(configs.enc_in, dropout=configs.dropout,
                                    current_level=configs.current_level,
                                    kernel_size=configs.kernel_size)
            self.projection_1 = nn.Conv1d(self.seq_len, self.seq_len + self.pred_len,
                                          kernel_size=1, stride=1, bias=False)
        else:
            self.sci_net_1 = SCINet(configs.enc_in, dropout=configs.dropout,
                                    current_level=configs.current_level,
                                    kernel_size=configs.kernel_size)
            self.sci_net_2 = SCINet(configs.enc_in, dropout=configs.dropout,
                                    current_level=configs.current_level,
                                    kernel_size=configs.kernel_size)
            self.projection_1 = nn.Conv1d(self.seq_len, self.pred_len,
                                          kernel_size=1, stride=1, bias=False)
            self.projection_2 = nn.Conv1d(self.seq_len + self.pred_len, self.seq_len + self.pred_len,
                                          kernel_size=1, bias=False)

        # For positional encoding
        self.pe_hidden_size = configs.enc_in
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1

        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # position-encoding
        pe = self.get_position_encoding(x_enc)
        if pe.shape[2] > x_enc.shape[2]:
            x_enc += pe[:, :, :-1]
        else:
            x_enc += self.get_position_encoding(x_enc)

        # SCINet
        dec_out = self.sci_net_1(x_enc)
        dec_out += x_enc
        dec_out = self.projection_1(dec_out)
        if self.num_stacks != 1:
            dec_out = torch.cat((x_enc, dec_out), dim=1)
            temp = dec_out
            dec_out = self.sci_net_2(dec_out)
            dec_out += temp
            dec_out = self.projection_2(dec_out)

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
        return signal


class Configs:
    def __init__(self, enc_in, dec_in, c_out, seq_len, pred_len, label_len,
                 d_model=512, d_ff=2048, d_layers=1,
                 n_heads=8, dropout=0.1, factor=3, activation='gelu',
                 embed='timeF', freq='h', dataset_type='ETTh1',
                 current_level=3, kernel_size=5):
        """
        初始化SCINet配置
        """
        self.task_name = 'long_term_forecast'
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out

        # 模型超参数
        self.d_model = 128
        self.d_ff = 256
        self.d_layers = d_layers
        self.n_heads = 8
        self.dropout = 0.1
        self.factor = 1
        self.activation = "gelu"
        self.embed = 'timeF'
        self.current_level = current_level  # SCINet tree depth
        self.kernel_size = kernel_size  # SCINet kernel size

        if dataset_type == 'M4-Daily':
            self.freq = 'd'
        elif dataset_type == 'M4-Weekly':
            self.freq = 'w'
        elif dataset_type == 'M4-Monthly':
            self.freq = 'm'
        elif dataset_type == 'M4-Quarterly':
            self.freq = 'q'
        elif dataset_type == 'M4-Yearly':
            self.freq = 'y'
        elif dataset_type == 'ETTh1':
            self.freq = 'h'
        elif dataset_type == 'ETTm1':
            self.freq = 't'
        elif dataset_type == 'Illness':
            self.freq = 'w'
        elif dataset_type == 'Exchange':
            self.freq = 'd'
        elif dataset_type == 'Weather':
            self.freq = 'n'
        else:
            self.freq = 'h'


def create_scinet_dataset(data, seq_len, label_len, pred_len):
    """创建SCINet数据集"""
    x_enc_list = []
    x_mark_enc_list = []
    x_dec_list = []
    x_mark_dec_list = []
    y_list = []

    total_len = len(data)
    total_seq_len = seq_len + pred_len

    for i in range(total_len - total_seq_len + 1):
        # encoder输入：前seq_len个时间步
        x_enc = data[i:i + seq_len]

        # decoder输入（SCINet不使用decoder，但为保持接口一致）
        x_dec = np.zeros((label_len + pred_len, data.shape[1]))

        # 目标值：预测的pred_len个时间步的目标变量
        y = data[i + seq_len:i + seq_len + pred_len, 0]

        # 时间特征占位符
        x_mark_enc = np.zeros((seq_len, 4))
        x_mark_dec = np.zeros((label_len + pred_len, 4))

        x_enc_list.append(x_enc)
        x_mark_enc_list.append(x_mark_enc)
        x_dec_list.append(x_dec)
        x_mark_dec_list.append(x_mark_dec)
        y_list.append(y)

    return (
        np.array(x_enc_list), np.array(x_mark_enc_list),
        np.array(x_dec_list), np.array(x_mark_dec_list),
        np.array(y_list)
    )


def train_scinet_model(X_train_enc, y_train, X_test_enc, y_test,
                       X_train_dec, X_test_dec,
                       configs, device='cuda',
                       num_epochs=100, batch_size=32, lr=0.0001):
    """训练SCINet模型"""
    x_enc_train = torch.FloatTensor(X_train_enc).to(device)
    x_dec_train = torch.FloatTensor(X_train_dec).to(device)
    x_mark_enc_train = torch.zeros(X_train_enc.shape[0], X_train_enc.shape[1], 4).to(device)
    x_mark_dec_train = torch.zeros(X_train_dec.shape[0], X_train_dec.shape[1], 4).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)

    model = SCINetModel(configs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    n_samples = len(x_enc_train)

    for epoch in range(num_epochs):
        total_loss = 0
        indices = np.random.permutation(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_x_enc = x_enc_train[batch_indices]
            batch_x_dec = x_dec_train[batch_indices]
            batch_x_mark_enc = x_mark_enc_train[batch_indices]
            batch_x_mark_dec = x_mark_dec_train[batch_indices]
            batch_y = y_train_tensor[batch_indices]

            optimizer.zero_grad()
            output = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            output_target = output[:, :, 0]
            loss = criterion(output_target, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(range(0, n_samples, batch_size)):.4f}')

    model.eval()
    with torch.no_grad():
        x_enc_test = torch.FloatTensor(X_test_enc).to(device)
        x_dec_test = torch.FloatTensor(X_test_dec).to(device)
        x_mark_enc_test = torch.zeros(X_test_enc.shape[0], X_test_enc.shape[1], 4).to(device)
        x_mark_dec_test = torch.zeros(X_test_dec.shape[0], X_test_dec.shape[1], 4).to(device)

        predictions = []
        for i in range(0, len(x_enc_test), batch_size):
            batch_x_enc = x_enc_test[i:i + batch_size]
            batch_x_dec = x_dec_test[i:i + batch_size]
            batch_x_mark_enc = x_mark_enc_test[i:i + batch_size]
            batch_x_mark_dec = x_mark_dec_test[i:i + batch_size]
            output = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            predictions.append(output[:, :, 0].cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

    return predictions


def train_evaluate_scinet(X_train_enc, y_train, X_val_enc, y_val,
                          X_train_dec, X_val_dec,
                          scaler, seq_len, label_len, pred_len,
                          feature_dim, params, dataset_type, device='cuda'):
    """训练并评估SCINet模型"""
    configs = Configs(
        enc_in=feature_dim,
        dec_in=feature_dim,
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_layers=params['d_layers'],
        dataset_type=dataset_type,
        current_level=params.get('current_level', 3),
        kernel_size=params.get('kernel_size', 5)
    )

    predictions = train_scinet_model(
        X_train_enc, y_train, X_val_enc, y_val,
        X_train_dec, X_val_dec,
        configs, device,
        num_epochs=params['num_epochs'],
        batch_size=params['batch_size'],
        lr=params['lr']
    )

    predictions_original = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_val_original = scaler.inverse_transform(y_val.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_val_original, predictions_original))
    mae = mean_absolute_error(y_val_original, predictions_original)
    return rmse, mae


# ==================== 数据集配置 ====================
extra_params = {
    "ETTh1": {"seq_len": 96, "label_len": 48, "pred_len": 96},
    "ETTm1": {"seq_len": 96, "label_len": 48, "pred_len": 96},
    "Illness": {"seq_len": 36, "label_len": 18, "pred_len": 24},
    "Exchange": {"seq_len": 96, "label_len": 48, "pred_len": 96},
    "Weather": {"seq_len": 96, "label_len": 48, "pred_len": 96}
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for dataset, columns in datasets.items():
    print('-' * 70)
    print(f"{dataset}: Processing...")

    target_column = columns["target_column"]
    base_path = "../../Datasets"

    extra_param = extra_params[dataset]
    seq_len = extra_param["seq_len"]
    label_len = extra_param["label_len"]
    pred_len = extra_param["pred_len"]

    dataset_param = params[dataset]
    scinet_param = dataset_param["scinet_param"]

    results = []

    # ==================== Process the clean data ====================
    input_clean_file = os.path.join(base_path, dataset, "clean.csv")
    clean_data = pd.read_csv(input_clean_file)
    numeric_columns = clean_data.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_columns:
        feature_columns = [target_column] + [col for col in numeric_columns if col != target_column]
    else:
        feature_columns = numeric_columns

    all_features = clean_data[feature_columns].values
    feature_dim = len(feature_columns)

    scalers = {}
    scaled_features = np.zeros_like(all_features)
    for i, col in enumerate(feature_columns):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features[:, i] = scaler.fit_transform(all_features[:, i].reshape(-1, 1)).flatten()
        scalers[col] = scaler

    x_enc, x_mark_enc, x_dec, x_mark_dec, y = create_scinet_dataset(
        scaled_features, seq_len, label_len, pred_len
    )

    # using best parameters
    split_idx = int(len(x_enc) * 0.7)
    x_enc_train = x_enc[:split_idx]
    x_enc_test = x_enc[split_idx:]
    x_dec_train = x_dec[:split_idx]
    x_dec_test = x_dec[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    configs = Configs(
        enc_in=feature_dim,
        dec_in=feature_dim,
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_layers=scinet_param['d_layers'],
        dataset_type=dataset,
        current_level=scinet_param.get('current_level', 3),
        kernel_size=scinet_param.get('kernel_size', 5)
    )

    print("Training final model on clean data with best parameters...")
    predictions_clean = train_scinet_model(
        x_enc_train, y_train, x_enc_test, y_test,
        x_dec_train, x_dec_test,
        configs, device,
        num_epochs=scinet_param['num_epochs'],
        batch_size=scinet_param['batch_size'],
        lr=scinet_param['lr']
    )

    target_scaler = scalers[target_column]
    predictions_clean_original = target_scaler.inverse_transform(predictions_clean.reshape(-1, 1))
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    clean_rmse = np.sqrt(mean_squared_error(y_test_original, predictions_clean_original))
    clean_mae = mean_absolute_error(y_test_original, predictions_clean_original)

    results.append(["clean.csv", clean_rmse, clean_mae, 0, 0])
    print("'clean.csv' is ok.")
    print(f"{clean_rmse}, {clean_mae}, 0, 0")

    # ==================== Process the dirty data ====================
    for portion in portion_list:
        for ingredient in Ingredients:
            for model in Imputation_Algorithms:
                for corr in Missing_rate:
                    input_dirty_file = os.path.join(base_path, dataset, "Mechanism", "timeseries", "decompose_replace",
                                                    f"{ingredient}", f"dirty-{ingredient}_{model}-{corr}.csv")

                    dirty_data = pd.read_csv(input_dirty_file)

                    if dirty_data.isnull().any().any():
                        dirty_data = dirty_data.fillna(0)

                    numeric_columns_dirty = dirty_data.select_dtypes(include=[np.number]).columns.tolist()
                    if target_column in numeric_columns_dirty:
                        feature_columns_dirty = [target_column] + [col for col in numeric_columns_dirty if col != target_column]
                    else:
                        feature_columns_dirty = numeric_columns_dirty

                    all_features_dirty = dirty_data[feature_columns_dirty].values

                    scaled_features_dirty = np.zeros_like(all_features_dirty)
                    for i, col in enumerate(feature_columns_dirty):
                        if col in scalers:
                            scaled_features_dirty[:, i] = scalers[col].transform(
                                all_features_dirty[:, i].reshape(-1, 1)).flatten()
                        else:
                            scaler = MinMaxScaler(feature_range=(0, 1))
                            scaled_features_dirty[:, i] = scaler.fit_transform(
                                all_features_dirty[:, i].reshape(-1, 1)).flatten()
                            scalers[col] = scaler

                    x_enc_dirty, x_mark_enc_dirty, x_dec_dirty, x_mark_dec_dirty, y_dirty = create_scinet_dataset(
                        scaled_features_dirty, seq_len, label_len, pred_len
                    )

                    if len(x_enc_dirty) == 0:
                        continue

                    x_enc_train_dirty = x_enc_dirty[:split_idx]
                    x_enc_test_dirty = x_enc_dirty[split_idx:]
                    x_dec_train_dirty = x_dec_dirty[:split_idx]
                    x_dec_test_dirty = x_dec_dirty[split_idx:]
                    y_train_dirty = y_dirty[:split_idx]
                    y_test_dirty = y_dirty[split_idx:]

                    predictions_dirty = train_scinet_model(
                        x_enc_train_dirty, y_train_dirty, x_enc_test_dirty, y_test_dirty,
                        x_dec_train_dirty, x_dec_test_dirty,
                        configs, device,
                        num_epochs=scinet_param['num_epochs'],
                        batch_size=scinet_param['batch_size'],
                        lr=scinet_param['lr']
                    )

                    predictions_dirty_original = target_scaler.inverse_transform(predictions_dirty.reshape(-1, 1))

                    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_dirty_original))
                    mae = mean_absolute_error(y_test_original, predictions_dirty_original)

                    pg_rmse = (rmse - clean_rmse) / clean_rmse if rmse > clean_rmse else 0
                    pg_mae = (mae - clean_mae) / clean_mae if mae > clean_mae else 0
                    results.append([f"dirty-{ingredient}_{model}-{corr}.csv", rmse, mae, pg_rmse, pg_mae])
                    print(f"'dirty-{ingredient}_{model}-{corr}.csv' is ok.")
                    print(f"{rmse}, {mae}, {pg_rmse}, {pg_mae}")

    output_dir = os.path.join("../../Downstream_Results", "timeseries", dataset)
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results, columns=["File Name", "RMSE", "MAE", "PG(RMSE)", "PG(MAE)"])
    results_df.to_csv(os.path.join(output_dir, f"scinet-decompose_replace-results-{dataset}1.csv"), index=False)

print("All tasks completed!")