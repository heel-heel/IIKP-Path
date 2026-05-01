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


datasets = {
    #"ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},
    #"ETTm1": {"target_column": "OT", "nonnumerical_column": "date"},
    #"Illness": {"target_column": "OT", "nonnumerical_column": "date"},
    #"Exchange": {"target_column": "OT", "nonnumerical_column": "date"},
    "Weather": {"target_column": "OT", "nonnumerical_column": "date"}
}
params = {
    "ETTh1":{
        "cnn_param":{
            "num_epochs":1,
            "batch_size":256,
            "lr":0.001
            }
    },
    "ETTm1":{
        "cnn_param":{
            "num_epochs":1,
            "batch_size":64,
            "lr":0.001
            }
    },
    "Illness":{
        "cnn_param":{
            "num_epochs":50,
            "batch_size":256,
            "lr":0.001
            }
    },
    "Exchange":{
        "cnn_param":{
            "num_epochs":1,
            "batch_size":64,
            "lr":0.0001
            }
    },
    "Weather":{
        "cnn_param":{
            "num_epochs":1,
            "batch_size":32,
            "lr":0.001
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

class CNNModel(nn.Module):
    """
    CNN模型用于时间序列预测
    使用1D卷积提取时间序列特征
    """

    def __init__(self, configs):
        super(CNNModel, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # 卷积层配置
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(in_channels=self.enc_in, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # 第二层卷积
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # 第三层卷积
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # 计算卷积后的序列长度
        self.conv_out_len = self._get_conv_output_length(self.seq_len)

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * self.conv_out_len, 512),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(256, self.pred_len * self.c_out)
        )

    def _get_conv_output_length(self, length):
        """计算经过卷积层后的输出长度"""
        # 经过两个MaxPool1d层，每个步长为2
        length = length // 2  # 第一层池化
        length = length // 2  # 第二层池化
        return length

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        前向传播
        x_enc: [batch_size, seq_len, enc_in]
        返回: [batch_size, pred_len, c_out]
        """
        # 转换维度: [batch_size, enc_in, seq_len] 以适应Conv1d
        x = x_enc.permute(0, 2, 1)

        # 卷积层
        x = self.conv_layers(x)

        # 展平
        x = x.reshape(x.size(0), -1)

        # 全连接层
        x = self.fc_layers(x)

        # 重塑为 [batch_size, pred_len, c_out]
        x = x.reshape(-1, self.pred_len, self.c_out)

        return x


class Configs:
    def __init__(self, enc_in, dec_in, c_out, seq_len, pred_len, label_len,
                 d_ff=2048,
                 n_heads=8, dropout=0.1, factor=3, activation='gelu',
                 embed='timeF', freq='h', dataset_type='ETTh1'):
        """
        初始化CNN配置

        参数说明：
        - enc_in: Encoder输入特征维度
        - dec_in: Decoder输入特征维度（CNN中不使用，但保留以保持接口一致）
        - c_out: 输出特征维度
        - seq_len: 输入序列长度
        - label_len: 标签长度（CNN中不使用）
        - pred_len: 预测长度
        - d_ff: 前馈网络维度（CNN中不使用）
        - n_heads: 多头注意力头数（CNN中不使用）
        - dropout: Dropout比率
        - factor: 注意力因子（CNN中不使用）
        - activation: 激活函数
        - embed: 时间特征编码类型（CNN中不使用）
        - freq: 数据频率（CNN中不使用）
        """
        self.task_name = 'long_term_forecast'
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out

        # 模型超参数
        self.d_ff = d_ff if d_ff else 512
        self.n_heads = n_heads
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


def create_cnn_dataset(data, seq_len, label_len, pred_len):
    """
    创建CNN数据集
    注意：CNN模型不需要decoder输入，这里保持接口与Transformer一致
    """
    x_enc_list = []
    y_list = []

    total_len = len(data)

    for i in range(total_len - seq_len - pred_len + 1):
        x_enc = data[i:i + seq_len]
        y = data[i + seq_len:i + seq_len + pred_len, 0]

        x_enc_list.append(x_enc)
        y_list.append(y)

    return (
        np.array(x_enc_list),  # x_enc
        None,  # x_mark_enc (not used in CNN)
        None,  # x_dec (not used in CNN)
        None,  # x_mark_dec (not used in CNN)
        np.array(y_list)  # y
    )


def train_cnn_model(X_train, y_train, X_test, y_test, configs,
                    device='cuda', num_epochs=100, batch_size=32, lr=0.0001):
    """训练CNN模型"""
    x_enc_train = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)

    model = CNNModel(configs).to(device)
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
            batch_y = y_train_tensor[batch_indices]

            optimizer.zero_grad()
            # CNN不需要x_mark_enc, x_dec, x_mark_dec，传入None
            output = model(batch_x_enc, None, None, None)
            output = output.squeeze(-1)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(range(0, n_samples, batch_size)):.4f}')

    # 评估
    model.eval()
    with torch.no_grad():
        x_enc_test = torch.FloatTensor(X_test).to(device)

        predictions = []
        for i in range(0, len(x_enc_test), batch_size):
            batch_x_enc = x_enc_test[i:i + batch_size]
            output = model(batch_x_enc, None, None, None)
            output = output.squeeze(-1)
            predictions.append(output.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

    return predictions


def train_evaluate_cnn(X_train_seq, y_train, X_val_seq, y_val, scaler,
                       seq_len, label_len, pred_len, feature_dim, params, dataset_type, device='cuda'):
    """训练并评估CNN模型"""
    configs = Configs(
        enc_in=feature_dim,
        dec_in=feature_dim,
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        dataset_type=dataset_type
    )

    predictions = train_cnn_model(
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
    cnn_param = dataset_param["cnn_param"]

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

    # 创建CNN数据集
    x_enc, _, _, _, y = create_cnn_dataset(
        scaled_features, seq_len, label_len, pred_len
    )

    # 使用最佳参数
    split_idx = int(len(x_enc) * 0.7)
    x_enc_train = x_enc[:split_idx]
    x_enc_test = x_enc[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    configs = Configs(
        enc_in=feature_dim,
        dec_in=feature_dim,
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        dataset_type=dataset
    )

    print("Training final model on clean data with best parameters...")
    predictions_clean = train_cnn_model(
        x_enc_train, y_train, x_enc_test, y_test, configs, device,
        num_epochs=cnn_param['num_epochs'],
        batch_size=cnn_param['batch_size'],
        lr=cnn_param['lr']
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
                    base_path, dataset, "Mechanism", "timeseries", "decompose_change_bad",
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

                x_enc_dirty, _, _, _, y_dirty = create_cnn_dataset(
                    scaled_features_dirty, seq_len, label_len, pred_len
                )

                if len(x_enc_dirty) == 0:
                    continue

                x_enc_train_dirty = x_enc_dirty[:split_idx]
                x_enc_test_dirty = x_enc_dirty[split_idx:]
                y_train_dirty = y_dirty[:split_idx]
                y_test_dirty = y_dirty[split_idx:]

                predictions_dirty = train_cnn_model(
                    x_enc_train_dirty, y_train_dirty, x_enc_test_dirty, y_test_dirty, configs, device,
                    num_epochs=cnn_param['num_epochs'],
                    batch_size=cnn_param['batch_size'],
                    lr=cnn_param['lr']
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
    results_df.to_csv(os.path.join(output_dir, f"cnn-decompose_change_bad-results-{dataset}1.csv"), index=False)

print("All tasks completed!")