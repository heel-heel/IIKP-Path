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
from layers.Conv_Blocks import Inception_Block_V1

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

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNetModel(nn.Module):
    def __init__(self, configs):
        super(TimesNetModel, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class Configs:
    def __init__(self, enc_in, dec_in, c_out, seq_len, pred_len, label_len,
                 d_model=512, d_ff=2048, e_layers=2,
                 n_heads=8, dropout=0.1, factor=3, activation='gelu',
                 embed='timeF', freq='h', dataset_type='ETTh1', top_k=5, num_kernels=6):
        """
        初始化TimesNet配置
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
        self.d_ff = 256
        self.e_layers = e_layers
        self.n_heads = 8
        self.dropout = 0.1
        self.factor = 1
        self.activation = "gelu"
        self.embed = 'timeF'
        self.top_k = top_k  # TimesNet参数
        self.num_kernels = num_kernels  # TimesNet参数

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


def create_timesnet_dataset(data, seq_len, label_len, pred_len):
    """创建TimesNet数据集"""
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

        # decoder输入（TimesNet不使用decoder，但为保持接口一致）
        x_dec = np.zeros((label_len + pred_len, data.shape[1]))

        # 目标值：预测的pred_len个时间步的目标变量
        y = data[i + seq_len:i + seq_len + pred_len, 0]

        # 时间特征占位符
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


def train_timesnet_model(X_train_enc, y_train, X_test_enc, y_test,
                         X_train_dec, X_test_dec,
                         configs, device='cuda',
                         num_epochs=100, batch_size=32, lr=0.0001):
    """训练TimesNet模型"""
    x_enc_train = torch.FloatTensor(X_train_enc).to(device)
    x_dec_train = torch.FloatTensor(X_train_dec).to(device)
    x_mark_enc_train = torch.zeros(X_train_enc.shape[0], X_train_enc.shape[1], 5).to(device)
    x_mark_dec_train = torch.zeros(X_train_dec.shape[0], X_train_dec.shape[1], 5).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)

    model = TimesNetModel(configs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    n_samples = len(x_enc_train)

    print(222)

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


def train_evaluate_timesnet(X_train_enc, y_train, X_val_enc, y_val,
                            X_train_dec, X_val_dec,
                            scaler, seq_len, label_len, pred_len,
                            feature_dim, params, dataset_type, device='cuda'):
    """训练并评估TimesNet模型"""
    configs = Configs(
        enc_in=feature_dim,
        dec_in=feature_dim,
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        e_layers=params['e_layers'],
        dataset_type=dataset_type,
        top_k=params.get('top_k', 5),
        num_kernels=params.get('num_kernels', 6)
    )

    print(111)

    predictions = train_timesnet_model(
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

timesnet_params = {
    "ETTh1": {"seq_len": 96, "label_len": 48, "pred_len": 96},
    "ETTm1": {"seq_len": 96, "label_len": 48, "pred_len": 96},
    "Illness": {"seq_len": 36, "label_len": 18, "pred_len": 24},
    "Exchange": {"seq_len": 96, "label_len": 48, "pred_len": 96},
    "Weather": {"seq_len": 96, "label_len": 48, "pred_len": 96}
}

Imputation_Algorithms = ['Mean', 'Median', 'Mode', 'KNN', 'HDI', 'MICE', 'IIM', 'SI',
                         'MFI', 'MissFI', 'XGBI', 'GAIN', 'MIDAE']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
Mechanism = ["MNAR"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

param_grid = {
    'e_layers': [4],
    'num_epochs': [1],
    'batch_size': [16],
    'lr': [0.001],
    'top_k': [2],
    'num_kernels': [1]
}

for dataset, columns in datasets.items():
    print('-' * 70)
    print(f"{dataset}: Processing with TimesNet...")

    target_column = columns["target_column"]
    base_path = "../../Datasets"

    timesnet_param = timesnet_params[dataset]
    seq_len = timesnet_param["seq_len"]
    label_len = timesnet_param["label_len"]
    pred_len = timesnet_param["pred_len"]

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

        x_enc, x_mark_enc, x_dec, x_mark_dec, y = create_timesnet_dataset(
            scaled_features, seq_len, label_len, pred_len
        )

        print("Performing 5-fold TimeSeries CV for parameter search on clean data...")
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
        #            'e_layers': random.choice(param_grid['e_layers']),
        #            'num_epochs': random.choice(param_grid['num_epochs']),
        #            'batch_size': random.choice(param_grid['batch_size']),
        #            'lr': random.choice(param_grid['lr']),
        #            'top_k': random.choice(param_grid['top_k']),
        #            'num_kernels': random.choice(param_grid['num_kernels'])
        #        }
        #        param_combinations.append(params)

        #    best_fold_rmse = float('inf')
        #    best_fold_mae = float('inf')
        #    best_fold_params = None

        #    for params in param_combinations:
        #        rmse, mae = train_evaluate_timesnet(
        #            X_train_enc, y_train, X_val_enc, y_val,
        #            X_train_dec, X_val_dec,
        #            scalers[target_column],
        #            seq_len, label_len, pred_len, feature_dim, params, dataset, device
        #        )

        #        print(f"  Testing params: e_layers={params['e_layers']}, "
        #              f"epochs={params['num_epochs']}, batch={params['batch_size']}, lr={params['lr']}, "
        #              f"top_k={params['top_k']}, num_kernels={params['num_kernels']}, "
        #              f"RMSE={rmse:.4f}, MAE={mae:.4f}")

        #        if rmse < best_fold_rmse:
        #            best_fold_rmse = rmse
        #            best_fold_mae = mae
        #            best_fold_params = params

        #    tscv_scores.append((best_fold_rmse, best_fold_mae, best_fold_params))
        #    print(f"  Fold best RMSE: {best_fold_rmse:.4f}, params: {best_fold_params}")

        #best_cv_params = min(tscv_scores, key=lambda x: x[0])[2]
        best_cv_params = {
            'e_layers': random.choice(param_grid['e_layers']),
            'num_epochs': random.choice(param_grid['num_epochs']),
            'batch_size': random.choice(param_grid['batch_size']),
            'lr': random.choice(param_grid['lr']),
            'top_k': random.choice(param_grid['top_k']),
            'num_kernels': random.choice(param_grid['num_kernels'])
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
            e_layers=best_cv_params['e_layers'],
            dataset_type=dataset,
            top_k=best_cv_params.get('top_k', 5),
            num_kernels=best_cv_params.get('num_kernels', 6)
        )

        print("Training final model on clean data with best parameters...")
        predictions_clean = train_timesnet_model(
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
        for rate in Missing_rate:
            input_dirty_file = os.path.join(base_path, dataset, "null", pattern, f"dirty-{rate}.csv")
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

            x_enc_dirty, x_mark_enc_dirty, x_dec_dirty, x_mark_dec_dirty, y_dirty = create_timesnet_dataset(
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

            predictions_dirty = train_timesnet_model(
                x_enc_train_dirty, y_train_dirty, x_enc_test_dirty, y_test_dirty,
                x_dec_train_dirty, x_dec_test_dirty,
                configs, device,
                num_epochs=best_cv_params['num_epochs'],
                batch_size=best_cv_params['batch_size'],
                lr=best_cv_params['lr']
            )

            predictions_dirty_original = target_scaler.inverse_transform(predictions_dirty.reshape(-1, 1))

            rmse = np.sqrt(mean_squared_error(y_test_original, predictions_dirty_original))
            mae = mean_absolute_error(y_test_original, predictions_dirty_original)

            pg_rmse = (rmse - clean_rmse) / clean_rmse if rmse > clean_rmse else 0
            pg_mae = (mae - clean_mae) / clean_mae if mae > clean_mae else 0

            results.append([f"dirty-{rate}.csv", rmse, mae, pg_rmse, pg_mae])
            print(f"dirty-{rate}.csv", rmse, mae, pg_rmse, pg_mae)

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

                x_enc_imp, x_mark_enc_imp, x_dec_imp, x_mark_dec_imp, y_imp = create_timesnet_dataset(
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

                predictions_imp = train_timesnet_model(
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
        output_results_file = os.path.join(output_results_path, f"timesnet-imputation-results-{dataset}.csv")
        results_df = pd.DataFrame(results, columns=["File Name", "RMSE", "MAE", "PG(RMSE)", "PG(MAE)"])
        results_df.to_csv(output_results_file, index=False)
        print(f"The results have saved to {output_results_file}.")

print("All tasks completed!")