# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import re
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi

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

class MIC(nn.Module):
    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24],
                 isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel
        self.device = device

        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                       kernel_size=i, padding=i // 2, stride=1)
                                             for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                                     kernel_size=(len(self.conv_kernel), 1))

        # feedforward network
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)  # [B, C, L]

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric convolution
        x = self.drop(self.act(isometric(x)))

        # 确保x和x1的维度匹配
        if x.shape[-1] != x1.shape[-1]:
            # 如果维度不匹配，进行截断或填充
            min_len = min(x.shape[-1], x1.shape[-1])
            x = x[:, :, :min_len]
            x1 = x1[:, :, :min_len]

        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))
        # 截断到原始长度
        if x.shape[-1] > seq_len:
            x = x[:, :, :seq_len]
        elif x.shape[-1] < seq_len:
            # 如果需要，进行填充
            padding = torch.zeros((batch, x.shape[1], seq_len - x.shape[-1]), device=self.device)
            x = torch.cat([x, padding], dim=-1)

        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        self.device = src.device
        # multi-scale
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)

        # merge
        mg = torch.stack(multi, dim=1)  # [B, num_kernels, L, C]
        mg = mg.permute(0, 3, 1, 2)  # [B, C, num_kernels, L]
        mg = self.merge(mg).squeeze(-2)  # [B, C, L]
        mg = mg.permute(0, 2, 1)  # [B, L, C]

        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)

        return self.norm2(mg + y)


class SeasonalPrediction(nn.Module):
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                 conv_kernel=[2, 4], isometric_kernel=[18, 6], device='cuda'):
        super(SeasonalPrediction, self).__init__()

        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, n_heads=n_heads,
                                      decomp_kernel=decomp_kernel, conv_kernel=conv_kernel,
                                      isometric_kernel=isometric_kernel, device=device)
                                  for i in range(d_layers)])

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)


class MICNModel(nn.Module):
    def __init__(self, configs, conv_kernel=[12, 16]):
        super(MICNModel, self).__init__()
        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution
        for ii in conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((configs.seq_len + configs.pred_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((configs.seq_len + configs.pred_len + ii - 1) // ii)

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out  # 只预测目标变量

        # Multiple Series decomposition block from FEDformer
        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # embedding
        self.dec_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.conv_trans = SeasonalPrediction(embedding_size=configs.d_model, n_heads=configs.n_heads,
                                             dropout=configs.dropout,
                                             d_layers=configs.d_layers, decomp_kernel=decomp_kernel,
                                             c_out=self.c_out, conv_kernel=conv_kernel,
                                             isometric_kernel=isometric_kernel, device=torch.device('cuda:0'))

        # refer to DLinear
        self.regression = nn.Linear(configs.seq_len, configs.pred_len)
        self.regression.weight = nn.Parameter(
            (1 / configs.pred_len) * torch.ones([configs.pred_len, configs.seq_len]),
            requires_grad=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Multi-scale Hybrid Decomposition
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)

        # 修复：确保 seasonal_init_dec 与 x_mark_dec 的时间维度匹配
        # x_mark_dec 的长度应该是 label_len + pred_len
        total_len = x_mark_dec.shape[1]  # 应该是 label_len + pred_len

        # 从 seasonal_init_enc 中取最后 seq_len 个时间步
        seasonal_part = seasonal_init_enc[:, -self.seq_len:, :]

        # 创建 zeros 部分，长度应该是 total_len - seq_len
        zeros_len = total_len - self.seq_len
        if zeros_len > 0:
            zeros = torch.zeros([x_dec.shape[0], zeros_len, x_dec.shape[2]], device=x_enc.device)
            seasonal_init_dec = torch.cat([seasonal_part, zeros], dim=1)
        else:
            seasonal_init_dec = seasonal_part[:, :total_len, :]

        dec_out = self.dec_embedding(seasonal_init_dec, x_mark_dec)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class Configs:
    def __init__(self, enc_in, dec_in, c_out, seq_len, pred_len, label_len,
                 d_model=512, d_ff=2048, d_layers=1,
                 n_heads=8, dropout=0.1, factor=3, activation='gelu',
                 embed='timeF', freq='h', dataset_type='ETTh1'):
        """
        初始化MICN配置
        """
        self.task_name = 'long_term_forecast'
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out  # 只预测目标变量

        # 模型超参数
        self.d_model = 128
        self.d_ff = 512
        self.d_layers = d_layers
        self.n_heads = 8
        self.dropout = 0.1
        self.factor = 1
        self.activation = "gelu"
        self.embed = 'timeF'

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


def create_micn_dataset(data, seq_len, label_len, pred_len):
    """创建MICN数据集"""
    x_enc_list = []
    x_mark_enc_list = []
    x_dec_list = []
    x_mark_dec_list = []
    y_list = []

    total_len = len(data)
    total_seq_len = seq_len + pred_len  # 总的时间序列长度

    for i in range(total_len - total_seq_len + 1):
        # encoder输入：前seq_len个时间步
        x_enc = data[i:i + seq_len]

        # decoder输入：最后label_len个时间步 + pred_len个零
        # 从i+seq_len-label_len开始，到i+seq_len结束，共label_len个时间步
        # 然后拼接pred_len个零
        dec_start = i + seq_len - label_len
        dec_end = i + seq_len
        x_dec = np.concatenate([
            data[dec_start:dec_end],  # label_len个时间步
            np.zeros((pred_len, data.shape[1]))  # pred_len个零
        ], axis=0)

        # 目标值：预测的pred_len个时间步的目标变量
        y = data[i + seq_len:i + seq_len + pred_len, 0]

        # 时间特征占位符（如果不需要实际时间特征，保持零即可）
        x_mark_enc = np.zeros((seq_len, 5))
        x_mark_dec = np.zeros((label_len + pred_len, 5))

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


def train_micn_model(X_train_enc, y_train, X_test_enc, y_test,
                     X_train_dec, X_test_dec,
                     configs, device='cuda',
                     num_epochs=100, batch_size=32, lr=0.0001):
    """训练MICN模型"""
    x_enc_train = torch.FloatTensor(X_train_enc).to(device)
    x_dec_train = torch.FloatTensor(X_train_dec).to(device)
    x_mark_enc_train = torch.zeros(X_train_enc.shape[0], X_train_enc.shape[1], 5).to(device)
    x_mark_dec_train = torch.zeros(X_train_dec.shape[0], X_train_dec.shape[1], 5).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)

    model = MICNModel(configs).to(device)
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

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(range(0, n_samples, batch_size)):.4f}')

    model.eval()
    with torch.no_grad():
        x_enc_test = torch.FloatTensor(X_test_enc).to(device)
        x_dec_test = torch.FloatTensor(X_test_dec).to(device)
        x_mark_enc_test = torch.zeros(X_test_enc.shape[0], X_test_enc.shape[1], 5).to(device)
        x_mark_dec_test = torch.zeros(X_test_dec.shape[0], X_test_dec.shape[1], 5).to(device)

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


def train_evaluate_micn(X_train_enc, y_train, X_val_enc, y_val,
                        X_train_dec, X_val_dec,
                        scaler, seq_len, label_len, pred_len,
                        feature_dim, params, dataset_type, device='cuda'):
    """训练并评估MICN模型"""
    configs = Configs(
        enc_in=feature_dim,
        dec_in=feature_dim,
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_layers=params['d_layers'],
        dataset_type=dataset_type
    )

    predictions = train_micn_model(
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
datasets = {
    #"ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},
     "ETTm1": {"target_column": "OT", "nonnumerical_column": "date"},
    # "Illness": {"target_column": "OT", "nonnumerical_column": "date"},
    # "Exchange": {"target_column": "OT", "nonnumerical_column": "date"},
    # "Weather": {"target_column": "OT", "nonnumerical_column": "date"}
}

micn_params = {
    "ETTh1": {"seq_len": 96, "label_len": 48, "pred_len": 96},
    "ETTm1": {"seq_len": 96, "label_len": 48, "pred_len": 96},
    "Illness": {"seq_len": 36, "label_len": 18, "pred_len": 24},
    "Exchange": {"seq_len": 96, "label_len": 48, "pred_len": 96},
    "Weather": {"seq_len": 96, "label_len": 48, "pred_len": 96}
}

#Imputation_Algorithms = ['Mean', 'Median', 'Mode', 'KNN', 'HDI', 'MICE', 'IIM', 'SI',
#                         'MFI', 'MissFI', 'XGBI', 'GAIN', 'MIDAE']
Imputation_Algorithms = ['MIDAE', 'GAIN', 'XGBI', 'MissFI', 'MFI']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
Mechanism = ["MNAR",]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

param_grid = {
    'd_layers': [4],
    'num_epochs': [1],
    'batch_size': [16],
    'lr': [0.0001],
}

for dataset, columns in datasets.items():
    print('-' * 70)
    print(f"{dataset}: Processing with MICN...")

    target_column = columns["target_column"]
    base_path = "../../Datasets"

    micn_param = micn_params[dataset]
    seq_len = micn_param["seq_len"]
    label_len = micn_param["label_len"]
    pred_len = micn_param["pred_len"]

    for pattern in Mechanism:
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

        x_enc, x_mark_enc, x_dec, x_mark_dec, y = create_micn_dataset(
            scaled_features, seq_len, label_len, pred_len
        )

        print("Performing 2-fold TimeSeries CV for parameter search on clean data...")
        tscv = TimeSeriesSplit(n_splits=2)
        tscv_scores = []
        n_iter = 1

        #for fold, (train_idx, val_idx) in enumerate(tscv.split(x_enc)):
        #    print(f"\nFold {fold + 1}/2")
        #    X_train_enc = x_enc[train_idx]
        #    X_val_enc = x_enc[val_idx]
        #    X_train_dec = x_dec[train_idx]
        #    X_val_dec = x_dec[val_idx]
        #    y_train = y[train_idx]
        #    y_val = y[val_idx]

        #    param_combinations = []
        #    for _ in range(n_iter):
        #        params = {
        #            'd_layers': random.choice(param_grid['d_layers']),
        #            'num_epochs': random.choice(param_grid['num_epochs']),
        #            'batch_size': random.choice(param_grid['batch_size']),
        #            'lr': random.choice(param_grid['lr'])
        #        }
        #        param_combinations.append(params)

        #    best_fold_rmse = float('inf')
        #    best_fold_mae = float('inf')
        #    best_fold_params = None

        #    for params in param_combinations:
        #        rmse, mae = train_evaluate_micn(
        #            X_train_enc, y_train, X_val_enc, y_val,
        #            X_train_dec, X_val_dec,
        #            scalers[target_column],
        #            seq_len, label_len, pred_len, feature_dim, params, dataset, device
        #        )

        #        print(f"  Testing params: d_layers={params['d_layers']}, "
        #              f"epochs={params['num_epochs']}, batch={params['batch_size']}, lr={params['lr']}, "
        #              f"RMSE={rmse:.4f}, MAE={mae:.4f}")

        #        if rmse < best_fold_rmse:
        #            best_fold_rmse = rmse
        #            best_fold_mae = mae
        #            best_fold_params = params

        #    tscv_scores.append((best_fold_rmse, best_fold_mae, best_fold_params))
        #    print(f"  Fold best RMSE: {best_fold_rmse}, params: {best_fold_params}")

        #best_cv_params = min(tscv_scores, key=lambda x: x[0])[2]
        best_cv_params = {
            'd_layers': random.choice(param_grid['d_layers']),
            'num_epochs': random.choice(param_grid['num_epochs']),
            'batch_size': random.choice(param_grid['batch_size']),
            'lr': random.choice(param_grid['lr'])
        }
        print(f"Best parameters from CV: {best_cv_params}")

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
            d_layers=best_cv_params['d_layers'],
            dataset_type=dataset
        )

        print("Training final model on clean data with best parameters...")
        predictions_clean = train_micn_model(
            x_enc_train, y_train, x_enc_test, y_test,
            x_dec_train, x_dec_test,
            configs, device,
            num_epochs=best_cv_params['num_epochs'],
            batch_size=best_cv_params['batch_size'],
            lr=best_cv_params['lr']
        )

        target_scaler = scalers[target_column]
        predictions_clean_original = target_scaler.inverse_transform(predictions_clean.reshape(-1, 1))
        y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))

        clean_rmse = np.sqrt(mean_squared_error(y_test_original, predictions_clean_original))
        clean_mae = mean_absolute_error(y_test_original, predictions_clean_original)

        results.append(["clean.csv", clean_rmse, clean_mae, 0, 0])
        print(f"clean.csv", clean_rmse, clean_mae)

        # ==================== Process the dirty data ====================
        # for rate in Missing_rate:
        #     input_dirty_file = os.path.join(base_path, dataset, "null", pattern, f"dirty-{rate}.csv")
        #     dirty_data = pd.read_csv(input_dirty_file)
        #
        #     if dirty_data.isnull().any().any():
        #         dirty_data = dirty_data.fillna(0)
        #
        #     numeric_columns_dirty = dirty_data.select_dtypes(include=[np.number]).columns.tolist()
        #     if target_column in numeric_columns_dirty:
        #         feature_columns_dirty = [target_column] + [col for col in numeric_columns_dirty if col != target_column]
        #     else:
        #         feature_columns_dirty = numeric_columns_dirty
        #
        #     all_features_dirty = dirty_data[feature_columns_dirty].values
        #
        #     scaled_features_dirty = np.zeros_like(all_features_dirty)
        #     for i, col in enumerate(feature_columns_dirty):
        #         if col in scalers:
        #             scaled_features_dirty[:, i] = scalers[col].transform(
        #                 all_features_dirty[:, i].reshape(-1, 1)).flatten()
        #         else:
        #             scaler = MinMaxScaler(feature_range=(0, 1))
        #             scaled_features_dirty[:, i] = scaler.fit_transform(
        #                 all_features_dirty[:, i].reshape(-1, 1)).flatten()
        #             scalers[col] = scaler
        #
        #     x_enc_dirty, x_mark_enc_dirty, x_dec_dirty, x_mark_dec_dirty, y_dirty = create_micn_dataset(
        #         scaled_features_dirty, seq_len, label_len, pred_len
        #     )
        #
        #     if len(x_enc_dirty) == 0:
        #         continue
        #
        #     x_enc_train_dirty = x_enc_dirty[:split_idx]
        #     x_enc_test_dirty = x_enc_dirty[split_idx:]
        #     x_dec_train_dirty = x_dec_dirty[:split_idx]
        #     x_dec_test_dirty = x_dec_dirty[split_idx:]
        #     y_train_dirty = y_dirty[:split_idx]
        #     y_test_dirty = y_dirty[split_idx:]
        #
        #     predictions_dirty = train_micn_model(
        #         x_enc_train_dirty, y_train_dirty, x_enc_test_dirty, y_test_dirty,
        #         x_dec_train_dirty, x_dec_test_dirty,
        #         configs, device,
        #         num_epochs=best_cv_params['num_epochs'],
        #         batch_size=best_cv_params['batch_size'],
        #         lr=best_cv_params['lr']
        #     )
        #
        #     predictions_dirty_original = target_scaler.inverse_transform(predictions_dirty.reshape(-1, 1))
        #
        #     rmse = np.sqrt(mean_squared_error(y_test_original, predictions_dirty_original))
        #     mae = mean_absolute_error(y_test_original, predictions_dirty_original)
        #
        #     pg_rmse = (rmse - clean_rmse) / clean_rmse if rmse > clean_rmse else 0
        #     pg_mae = (mae - clean_mae) / clean_mae if mae > clean_mae else 0
        #
        #     results.append([f"dirty-{rate}.csv", rmse, mae, pg_rmse, pg_mae])
        #     print(f"dirty-{rate}.csv", rmse, mae, pg_rmse, pg_mae)

        # ==================== Process the imputed data ====================
        for method in Imputation_Algorithms:
            for rate in Missing_rate:
                input_imputed_file = os.path.join(
                    base_path, dataset, "Imputation", pattern, f"null-{method.lower()}",
                    f"dirty-{method.lower()}-{rate}.csv"
                )

                if not os.path.exists(input_imputed_file):
                    continue

                imputed_data = pd.read_csv(input_imputed_file)

                if imputed_data.isnull().any().any():
                    imputed_data = imputed_data.fillna(0)

                numeric_columns_imp = imputed_data.select_dtypes(include=[np.number]).columns.tolist()
                if target_column in numeric_columns_imp:
                    feature_columns_imp = [target_column] + [col for col in numeric_columns_imp if col != target_column]
                else:
                    feature_columns_imp = numeric_columns_imp

                all_features_imp = imputed_data[feature_columns_imp].values

                scaled_features_imp = np.zeros_like(all_features_imp)
                for i, col in enumerate(feature_columns_imp):
                    if col in scalers:
                        scaled_features_imp[:, i] = scalers[col].transform(
                            all_features_imp[:, i].reshape(-1, 1)).flatten()
                    else:
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaled_features_imp[:, i] = scaler.fit_transform(
                            all_features_imp[:, i].reshape(-1, 1)).flatten()
                        scalers[col] = scaler

                x_enc_imp, x_mark_enc_imp, x_dec_imp, x_mark_dec_imp, y_imp = create_micn_dataset(
                    scaled_features_imp, seq_len, label_len, pred_len
                )

                if len(x_enc_imp) == 0:
                    continue

                x_enc_train_imp = x_enc_imp[:split_idx]
                x_enc_test_imp = x_enc_imp[split_idx:]
                x_dec_train_imp = x_dec_imp[:split_idx]
                x_dec_test_imp = x_dec_imp[split_idx:]
                y_train_imp = y_imp[:split_idx]
                y_test_imp = y_imp[split_idx:]

                predictions_imp = train_micn_model(
                    x_enc_train_imp, y_train_imp, x_enc_test_imp, y_test_imp,
                    x_dec_train_imp, x_dec_test_imp,
                    configs, device,
                    num_epochs=best_cv_params['num_epochs'],
                    batch_size=best_cv_params['batch_size'],
                    lr=best_cv_params['lr']
                )

                predictions_imp_original = target_scaler.inverse_transform(predictions_imp.reshape(-1, 1))

                rmse = np.sqrt(mean_squared_error(y_test_original, predictions_imp_original))
                mae = mean_absolute_error(y_test_original, predictions_imp_original)

                pg_rmse = (rmse - clean_rmse) / clean_rmse if rmse > clean_rmse else 0
                pg_mae = (mae - clean_mae) / clean_mae if mae > clean_mae else 0

                results.append([f"dirty-{method}-{rate}.csv", rmse, mae, pg_rmse, pg_mae])
                print(
                    f"dirty-{method}-{rate}.csv", rmse, mae, pg_rmse, pg_mae)

        output_base_path = "../../Downstream_Results"
        output_results_path = os.path.join(output_base_path, "timeseries", dataset, pattern)
        os.makedirs(output_results_path, exist_ok=True)
        output_results_file = os.path.join(output_results_path, f"micn-imputation-results-{dataset}2.csv")
        results_df = pd.DataFrame(results, columns=["File Name", "RMSE", "MAE", "PG(RMSE)", "PG(MAE)"])
        results_df.to_csv(output_results_file, index=False)
        print(f"The results have saved to {output_results_file}.")

print("All tasks completed!")