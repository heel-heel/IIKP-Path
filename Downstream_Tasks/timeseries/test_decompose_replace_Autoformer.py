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
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from layers.Embed import DataEmbedding_wo_pos
from sklearn.model_selection import TimeSeriesSplit


datasets = {
    #"ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},
    #"ETTm1": {"target_column": "OT", "nonnumerical_column": "date"},
    #"Illness": {"target_column": "OT", "nonnumerical_column": "date"},
    #"Exchange": {"target_column": "OT", "nonnumerical_column": "date"},
    "Weather": {"target_column": "OT", "nonnumerical_column": "date"}
}
params = {
    "ETTh1":{
        "autoformer_param":{
            "e_layers":3,
            "d_layers":2,
            "num_epochs":2,
            "batch_size":32,
            "lr":0.0001,
            "moving_avg":25
            }
    },
    "ETTm1":{
        "autoformer_param":{
            "e_layers":3,
            "d_layers":2,
            "num_epochs":1,
            "batch_size":64,
            "lr":0.001,
            "moving_avg":25
            }
    },
    "Illness":{
        "autoformer_param":{
            "e_layers":2,
            "d_layers":2,
            "num_epochs":20,
            "batch_size":32,
            "lr":0.0001,
            "moving_avg":25
            }
    },
    "Exchange":{
        "autoformer_param":{
            "e_layers":4,
            "d_layers":3,
            "num_epochs":1,
            "batch_size":32,
            "lr":0.001,
            "moving_avg":25
            }
    },
    "Weather":{
        "autoformer_param":{
            "e_layers":3,
            "d_layers":3,
            "num_epochs":1,
            "batch_size":32,
            "lr":0.001,
            "moving_avg":25
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

class AutoformerModel(nn.Module):
    def __init__(self, configs):
        super(AutoformerModel, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class Configs:
    def __init__(self, enc_in, dec_in, c_out, seq_len, pred_len, label_len,
                 d_model=512, d_ff=2048, e_layers=2, d_layers=1,
                 n_heads=8, dropout=0.1, factor=3, activation='gelu',
                 embed='timeF', freq='h', dataset_type='ETTh1', moving_avg=25):
        """
        初始化Autoformer配置

        参数说明：
        - enc_in: Encoder输入特征维度
        - dec_in: Decoder输入特征维度
        - c_out: 输出特征维度
        - seq_len: 输入序列长度
        - label_len: 标签长度
        - pred_len: 预测长度
        - d_model: 模型隐藏层维度
        - d_ff: 前馈网络维度
        - e_layers: Encoder层数
        - d_layers: Decoder层数
        - n_heads: 多头注意力头数
        - dropout: Dropout比率
        - factor: 注意力因子
        - activation: 激活函数
        - embed: 时间特征编码类型
        - freq: 数据频率
        - moving_avg: 移动平均窗口大小
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
        self.d_ff = 512
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.n_heads = 8
        self.dropout = 0.1
        self.factor = 1
        self.activation = "gelu"
        self.embed = 'timeF'
        self.moving_avg = moving_avg  # Autoformer特有参数

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


def create_autoformer_dataset(data, seq_len, label_len, pred_len):
    """创建Autoformer数据集"""
    x_enc_list = []
    x_mark_enc_list = []
    x_dec_list = []
    x_mark_dec_list = []
    y_list = []

    total_len = len(data)

    for i in range(total_len - seq_len - pred_len + 1):
        x_enc = data[i:i + seq_len]
        x_dec = np.concatenate([
            data[i + seq_len - label_len:i + seq_len],
            np.zeros((pred_len, data.shape[1]))
        ], axis=0)
        y = data[i + seq_len:i + seq_len + pred_len, 0]
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


def train_autoformer_model(X_train, y_train, X_test, y_test, configs,
                           device='cuda', num_epochs=100, batch_size=32, lr=0.0001):
    """训练Autoformer模型"""
    x_enc_train = torch.FloatTensor(X_train).to(device)
    x_dec_train = torch.FloatTensor(X_train).to(device)
    x_mark_enc_train = torch.zeros(X_train.shape[0], X_train.shape[1], 4).to(device)

    label_len = configs.label_len
    pred_len = configs.pred_len
    dec_len = label_len + pred_len
    x_mark_dec_train = torch.zeros(X_train.shape[0], dec_len, 4).to(device)
    #x_mark_dec_train = torch.zeros(X_train.shape[0], X_train.shape[1], 4).to(device)

    y_train_tensor = torch.FloatTensor(y_train).to(device)

    model = AutoformerModel(configs).to(device)
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
        x_enc_test = torch.FloatTensor(X_test).to(device)
        x_dec_test = torch.FloatTensor(X_test).to(device)
        x_mark_enc_test = torch.zeros(X_test.shape[0], X_test.shape[1], 4).to(device)
        #x_mark_dec_test = torch.zeros(X_test.shape[0], X_test.shape[1], 4).to(device)
        x_mark_dec_test = torch.zeros(X_test.shape[0], dec_len, 4).to(device)

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


def train_evaluate_autoformer(X_train_seq, y_train, X_val_seq, y_val, scaler,
                              seq_len, label_len, pred_len, feature_dim, params, dataset_type, device='cuda'):
    """训练和评估Autoformer模型"""
    configs = Configs(
        enc_in=feature_dim,
        dec_in=feature_dim,
        c_out=feature_dim,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        e_layers=params['e_layers'],
        d_layers=params['d_layers'],
        dataset_type=dataset_type,
        moving_avg=params.get('moving_avg', 25)  # Autoformer特有参数
    )

    predictions = train_autoformer_model(
        X_train_seq, y_train, X_val_seq, y_val, configs, device,
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
    autoformer_param = dataset_param["autoformer_param"]

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

    x_enc, x_mark_enc, x_dec, x_mark_dec, y = create_autoformer_dataset(
        scaled_features, seq_len, label_len, pred_len
    )

    # using best parameters
    split_idx = int(len(x_enc) * 0.7)
    x_enc_train = x_enc[:split_idx]
    x_enc_test = x_enc[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    configs = Configs(
        enc_in=feature_dim,
        dec_in=feature_dim,
        c_out=feature_dim,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        e_layers=autoformer_param['e_layers'],
        d_layers=autoformer_param['d_layers'],
        moving_avg=autoformer_param['moving_avg']
    )

    predictions_clean = train_autoformer_model(
        x_enc_train, y_train, x_enc_test, y_test, configs, device,
        num_epochs=autoformer_param['num_epochs'],
        batch_size=autoformer_param['batch_size'],
        lr=autoformer_param['lr']
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

                    x_enc_dirty, x_mark_enc_dirty, x_dec_dirty, x_mark_dec_dirty, y_dirty = create_autoformer_dataset(
                        scaled_features_dirty, seq_len, label_len, pred_len
                    )

                    if len(x_enc_dirty) == 0:
                        continue

                    x_enc_train_dirty = x_enc_dirty[:split_idx]
                    x_enc_test_dirty = x_enc_dirty[split_idx:]
                    y_train_dirty = y_dirty[:split_idx]
                    y_test_dirty = y_dirty[split_idx:]

                    predictions_dirty = train_autoformer_model(
                        x_enc_train_dirty, y_train_dirty, x_enc_test_dirty, y_test_dirty, configs, device,
                        num_epochs=autoformer_param['num_epochs'],
                        batch_size=autoformer_param['batch_size'],
                        lr=autoformer_param['lr']
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
    results_df.to_csv(os.path.join(output_dir, f"autoformer-decompose_replace-results-{dataset}1.csv"), index=False)

print("All tasks completed!")