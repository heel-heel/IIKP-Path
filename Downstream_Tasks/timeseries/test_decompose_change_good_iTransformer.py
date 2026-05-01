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
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
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
        "itransformer_param":{
            "e_layers":4,
            "num_epochs":2,
            "batch_size":64,
            "lr":0.0001,
            "d_model":256,
            "d_ff":256,
            "factor":4
            }
    },
    "ETTm1":{
        "itransformer_param":{
            "e_layers":2,
            "num_epochs":1,
            "batch_size":32,
            "lr":0.0001,
            "d_model":128,
            "d_ff":512,
            "factor":4
            }
    },
    "Illness":{
        "itransformer_param":{
            "e_layers":4,
            "num_epochs":10,
            "batch_size":64,
            "lr":0.001,
            "d_model":128,
            "d_ff":512,
            "factor":4
            }
    },
    "Exchange":{
        "itransformer_param":{
            "e_layers":3,
            "num_epochs":1,
            "batch_size":64,
            "lr":0.0001,
            "d_model":128,
            "d_ff":512,
            "factor":4
            }
    },
    "Weather":{
        "itransformer_param":{
            "e_layers":4,
            "num_epochs":1,
            "batch_size":32,
            "lr":0.0001,
            "d_model":128,
            "d_ff":512,
            "factor":4
            }
    },
}

Ingredients = ['resid', 'trend', 'seasonal']
portion_list = [50]
#corr_list = [70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]
corr_list = [70, 98]


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

class iTransformerModel(nn.Module):
    def __init__(self, configs):
        super(iTransformerModel, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Embedding (倒置嵌入)
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Projection
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class Configs:
    def __init__(self, enc_in, dec_in, c_out, seq_len, pred_len,
                 d_model=512, d_ff=2048, e_layers=2, d_layers=1,
                 n_heads=8, dropout=0.1, factor=3, activation='gelu',
                 embed='timeF', freq='h', dataset_type='ETTh1'):
        """
        初始化iTransformer配置

        参数说明：
        - enc_in: Encoder输入特征维度
        - dec_in: Decoder输入特征维度（iTransformer不使用）
        - c_out: 输出特征维度
        - seq_len: 输入序列长度
        - pred_len: 预测长度
        - d_model: 模型隐藏层维度
        - d_ff: 前馈网络维度
        - e_layers: Encoder层数
        - d_layers: Decoder层数（iTransformer不使用）
        - n_heads: 多头注意力头数
        - dropout: Dropout比率
        - factor: 注意力因子
        - activation: 激活函数
        - embed: 时间特征编码类型
        - freq: 数据频率
        """
        self.task_name = 'long_term_forecast'
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out

        # 模型超参数
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.factor = factor
        self.activation = activation
        self.embed = embed

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


def create_itransformer_dataset(data, seq_len, pred_len):
    """创建iTransformer数据集（只需要x_enc，不需要x_dec）"""
    x_enc_list = []
    x_mark_enc_list = []
    y_list = []

    total_len = len(data)

    for i in range(total_len - seq_len - pred_len + 1):
        x_enc = data[i:i + seq_len]
        y = data[i + seq_len:i + seq_len + pred_len, 0]  # 只预测目标变量
        x_mark_enc = np.zeros((seq_len, 4))

        x_enc_list.append(x_enc)
        x_mark_enc_list.append(x_mark_enc)
        y_list.append(y)

    return (
        np.array(x_enc_list), np.array(x_mark_enc_list),
        np.array(y_list)
    )


def train_itransformer_model(x_enc_train, y_train, x_enc_test, y_test, configs,
                             device='cuda', num_epochs=100, batch_size=32, lr=0.0001):
    """训练iTransformer模型"""
    x_enc_train = torch.FloatTensor(x_enc_train).to(device)
    x_enc_test = torch.FloatTensor(x_enc_test).to(device)

    x_mark_enc_train = torch.zeros(x_enc_train.shape[0], x_enc_train.shape[1], 4).to(device)
    x_mark_enc_test = torch.zeros(x_enc_test.shape[0], x_enc_test.shape[1], 4).to(device)

    y_train_tensor = torch.FloatTensor(y_train).to(device)

    model = iTransformerModel(configs).to(device)
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
            batch_x_mark_enc = x_mark_enc_train[batch_indices]
            batch_y = y_train_tensor[batch_indices]

            optimizer.zero_grad()
            # iTransformer只需要x_enc和x_mark_enc，x_dec和x_mark_dec可以传None
            output = model(batch_x_enc, batch_x_mark_enc, None, None)
            output_target = output[:, :, 0]
            loss = criterion(output_target, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            #print(i)

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(range(0, n_samples, batch_size))
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 预测
    model.eval()
    with torch.no_grad():
        predictions = []
        for i in range(0, len(x_enc_test), batch_size):
            batch_x_enc = x_enc_test[i:i + batch_size]
            batch_x_mark_enc = x_mark_enc_test[i:i + batch_size]
            output = model(batch_x_enc, batch_x_mark_enc, None, None)
            predictions.append(output[:, :, 0].cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

    return predictions


def train_evaluate_itransformer(X_train_seq, y_train, X_val_seq, y_val, scaler,
                                seq_len, pred_len, feature_dim, params, dataset_type, device='cuda'):
    """训练和评估iTransformer模型"""
    configs = Configs(
        enc_in=feature_dim,
        dec_in=feature_dim,
        c_out=feature_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=params.get('d_model', 128),
        d_ff=params.get('d_ff', 512),
        e_layers=params['e_layers'],
        d_layers=params.get('d_layers', 1),
        n_heads=params.get('n_heads', 8),
        dropout=params.get('dropout', 0.1),
        factor=params.get('factor', 3),
        activation=params.get('activation', 'gelu'),
        dataset_type=dataset_type
    )

    predictions = train_itransformer_model(
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
    "ETTh1": {"seq_len": 96, "pred_len": 96},
    "ETTm1": {"seq_len": 96, "pred_len": 96},
    "Illness": {"seq_len": 36, "pred_len": 24},
    "Exchange": {"seq_len": 96, "pred_len": 96},
    "Weather": {"seq_len": 96, "pred_len": 96}
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for dataset, columns in datasets.items():
    print('-' * 70)
    print(f"{dataset}: Processing...")

    target_column = columns["target_column"]
    base_path = "../../Datasets"

    extra_param = extra_params[dataset]
    seq_len = extra_param["seq_len"]
    pred_len = extra_param["pred_len"]

    dataset_param = params[dataset]
    itransformer_param = dataset_param["itransformer_param"]

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

    # iTransformer只需要x_enc，不需要x_dec
    x_enc, x_mark_enc, y = create_itransformer_dataset(
        scaled_features, seq_len, pred_len
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
        pred_len=pred_len,
        d_model=itransformer_param.get('d_model', 128),
        d_ff=itransformer_param.get('d_ff', 512),
        e_layers=itransformer_param['e_layers'],
        n_heads=itransformer_param.get('n_heads', 8),
        dropout=itransformer_param.get('dropout', 0.1),
        factor=itransformer_param.get('factor', 3),
        dataset_type=dataset
    )

    print("Training final model on clean data with best parameters...")
    predictions_clean = train_itransformer_model(
        x_enc_train, y_train, x_enc_test, y_test, configs, device,
        num_epochs=itransformer_param['num_epochs'],
        batch_size=itransformer_param['batch_size'],
        lr=itransformer_param['lr']
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
        for corr in corr_list:
            for ingredient in Ingredients:
                input_dirty_file = os.path.join(
                    base_path, dataset, "Mechanism", "timeseries", "decompose_change_good",
                    f"{ingredient}", f"dirty-{ingredient}-{portion}-{corr}.csv"
                )

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

                x_enc_dirty, x_mark_enc_dirty, y_dirty = create_itransformer_dataset(
                    scaled_features_dirty, seq_len, pred_len
                )

                if len(x_enc_dirty) == 0:
                    continue

                x_enc_train_dirty = x_enc_dirty[:split_idx]
                x_enc_test_dirty = x_enc_dirty[split_idx:]
                y_train_dirty = y_dirty[:split_idx]
                y_test_dirty = y_dirty[split_idx:]

                predictions_dirty = train_itransformer_model(
                    x_enc_train_dirty, y_train_dirty, x_enc_test_dirty, y_test_dirty, configs, device,
                    num_epochs=itransformer_param['num_epochs'],
                    batch_size=itransformer_param['batch_size'],
                    lr=itransformer_param['lr']
                )

                predictions_dirty_original = target_scaler.inverse_transform(predictions_dirty.reshape(-1, 1))

                rmse = np.sqrt(mean_squared_error(y_test_original, predictions_dirty_original))
                mae = mean_absolute_error(y_test_original, predictions_dirty_original)

                pg_rmse = (rmse - clean_rmse) / clean_rmse if rmse > clean_rmse else 0
                pg_mae = (mae - clean_mae) / clean_mae if mae > clean_mae else 0
                results.append([f"dirty-{ingredient}-{portion}-{corr}.csv", rmse, mae, pg_rmse, pg_mae])
                print(f"'dirty-{ingredient}-{portion}-{corr}.csv' is ok.")
                print(f"{rmse}, {mae}, {pg_rmse}, {pg_mae}")

    output_dir = os.path.join("../../Downstream_Results", "timeseries", dataset)
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results, columns=["File Name", "RMSE", "MAE", "PG(RMSE)", "PG(MAE)"])
    results_df.to_csv(os.path.join(output_dir, f"itransformer-decompose_change_good-results-{dataset}1.csv"), index=False)

print("All tasks completed!")