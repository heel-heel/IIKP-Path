# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

class CNNRegressor(nn.Module):
    """CNN模型用于回归任务"""

    def __init__(self, input_dim, conv_channels=[64, 128], kernel_sizes=[3],
                 fc_units=128, dropout=0.1):
        super(CNNRegressor, self).__init__()

        # 确保kernel_sizes是列表
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]

        # 构建卷积层
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # 初始输入通道数（将特征视为单通道）

        for i, out_channels in enumerate(conv_channels):
            # 选择当前层的卷积核大小（循环使用kernel_sizes）
            kernel_size = kernel_sizes[i % len(kernel_sizes)]
            padding = kernel_size // 2  # 保持维度

            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels

        # 计算卷积后的输出维度
        self.conv_output_dim = self._get_conv_output_dim(input_dim, conv_channels, kernel_sizes)

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_dim, fc_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_units, fc_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_units // 2, 1)  # 回归任务输出1个值
        )

    def _get_conv_output_dim(self, input_dim, conv_channels, kernel_sizes):
        """计算经过卷积层后的输出维度"""
        length = input_dim

        for i in range(len(conv_channels)):
            # 经过MaxPool1d后长度减半
            length = length // 2

        # 最终输出维度 = 最后一层通道数 * 长度
        return conv_channels[-1] * max(length, 1)

    def forward(self, x):
        """
        前向传播
        x: [batch_size, input_dim]
        返回: [batch_size]
        """
        # 添加通道维度: [batch_size, 1, input_dim]
        x = x.unsqueeze(1)

        # 通过卷积层
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # 展平: [batch_size, -1]
        x = x.reshape(x.size(0), -1)

        # 通过全连接层
        x = self.fc_layers(x)

        return x.squeeze()  # 返回 [batch_size]


# Scikit-learn compatible wrapper for CNN Regressor
class CNNRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=None, conv_channels=[64, 128], kernel_sizes=[3],
                 fc_units=128, dropout=0.1, learning_rate=0.001,
                 batch_size=64, epochs=200, random_state=42, device='cpu'):
        self.input_dim = input_dim
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.fc_units = fc_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.scaler = StandardScaler()

    def _create_model(self, input_dim):
        return CNNRegressor(
            input_dim=input_dim,
            conv_channels=self.conv_channels,
            kernel_sizes=self.kernel_sizes,
            fc_units=self.fc_units,
            dropout=self.dropout
        )

    def fit(self, X, y):
        # 设置随机种子
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # 保存输入维度
        if self.input_dim is None:
            self.input_dim = X.shape[1]

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X.astype(np.float32))

        # 转换为张量
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).reshape(-1, 1).to(self.device)

        # 创建模型
        self.model = self._create_model(self.input_dim).to(self.device)

        # 创建数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 训练
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y.squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {total_loss / len(dataloader):.4f}')

        return self

    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X.astype(np.float32))
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy()

    def get_params(self, deep=True):
        return {
            'conv_channels': self.conv_channels,
            'kernel_sizes': self.kernel_sizes,
            'fc_units': self.fc_units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def score(self, X, y):
        """返回负的MAE（sklearn期望更高的分数）"""
        y_pred = self.predict(X)
        return -mean_absolute_error(y, y_pred)


def get_best_cnn_params_with_cv(X_train, y_train, n_folds=2, n_iter=1):
    param_dist = {
        'conv_channels': [[64, 128]],
        'kernel_sizes': [[5]],
        'fc_units': [64],
        'dropout': [0.1],
        'learning_rate': [0.01],
        'batch_size': [32],
        'epochs': [1]
    }

    # 创建包装器模型
    cnn_reg = CNNRegressorWrapper(input_dim=X_train.shape[1], random_state=42)

    # 创建评分器（使用负MAE，因为sklearn期望最大化分数）
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # 创建RandomizedSearchCV对象
    random_search = RandomizedSearchCV(
        cnn_reg,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=n_folds,
        scoring=scorer,
        n_jobs=-1,
        verbose=10,
        random_state=42
    )

    # 执行搜索
    random_search.fit(X_train, y_train)
    return random_search.best_params_


def cnn_regressor(X_train, X_test, y_train, y_test, best_params=None):
    """使用CNN进行回归"""
    # 创建模型
    if best_params is None:
        model = CNNRegressorWrapper(input_dim=X_train.shape[1], random_state=42)
    else:
        model = CNNRegressorWrapper(
            input_dim=X_train.shape[1],
            **best_params,
            random_state=42
        )

    # 训练和预测
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse, mae = evaluate(y_test, y_pred)
    return mse, mae


def testing_func(rep_df, clean_df, target, feature_schema, best_params=None):
    """测试函数"""
    feature_schema = [x.lower() for x in feature_schema]
    rep_df.columns = rep_df.columns.str.lower()
    clean_df.columns = clean_df.columns.str.lower()
    target = target.lower()

    df_encoded = pd.get_dummies(rep_df[feature_schema])
    sanitized_feature_names = {}
    for feature_names_str in df_encoded.columns:
        valid_chars = [char for char in feature_names_str if char not in ['[', ']', '<']]
        sanitized_feature_names[feature_names_str] = ''.join(valid_chars)

    for df in [rep_df, clean_df]:
        df[target] = df[target].astype(str).str.strip('%')
        df.loc[df[target] == 'empty', target] = 0
        df.loc[pd.isna(df[target]), target] = 0
        df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0)
        for i in range(len(df[target])):
            try:
                df.loc[i, target] = float(df.loc[i, target])
            except:
                df.loc[i, target] = 0

    df_encoded.rename(columns=sanitized_feature_names, inplace=True)
    train_indices, test_indices = train_test_split(range(len(df_encoded)), test_size=0.2, random_state=0)
    X_train = df_encoded.iloc[train_indices]
    y_train = rep_df[target].iloc[train_indices]
    X_test = df_encoded.iloc[test_indices]
    y_test = clean_df[target].iloc[test_indices]

    res_dict = {}
    mse, mae = cnn_regressor(X_train, X_test, y_train, y_test, best_params)
    res_dict['cnn'] = [mse, mae]
    return res_dict


def evaluate(y_test, y_pred):
    """评估函数"""
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae


if __name__ == "__main__":
    input_base_path = "../../Datasets"
    output_base_path = "../../Downstream_Results"

    datasets = {
        "concrete": {"target_column": "concrete_compressive_strength", "nonnumerical_column": "None"},
        # "CCPP": {"target_column": "PE", "nonnumerical_column": "None"},
        # "AirfoilSelfNoise": {"target_column": "SSPL", "nonnumerical_column": "None"},
        # "Abalone": {"target_column": "Rings", "nonnumerical_column": "None"},
        # "ParisHousing": {"target_column": "price", "nonnumerical_column": "None"},
    }

    Imputation_Algorithms = [#'mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim',
         'si',
        #'mfi', 'missfi', 'xgbi',
                             #'gain', 'midae'
         ]
    Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    Mechanism = ["MAR", "MNAR"]

    for dataset, columns in datasets.items():
        print('-' * 70)
        print(f"{dataset}: Processing with CNN...")
        target = columns["target_column"]
        clean_path = os.path.join(input_base_path, dataset, 'clean.csv')
        clean_df = pd.read_csv(clean_path).astype(str)
        clean_df.fillna('nan', inplace=True)
        feature_schema = list(clean_df.columns)
        feature_schema.remove(target)

        # 在干净数据上执行带交叉验证的随机搜索
        print("Performing grid search on clean data...")
        df_encoded = pd.get_dummies(clean_df[feature_schema])
        train_indices, test_indices = train_test_split(range(len(df_encoded)), test_size=0.2, random_state=0)
        X_train = df_encoded.iloc[train_indices].values.astype(np.float32)
        y_train = clean_df[target].iloc[train_indices].values.astype(np.float32)

        # 执行参数搜索（可以使用简化版进行快速测试）
        # best_params = get_best_cnn_params_with_cv(X_train, y_train, n_folds=5, n_iter=50)
        best_params = get_best_cnn_params_with_cv(X_train, y_train, n_folds=2, n_iter=1)
        print(f"best parameters: {best_params}")

        for pattern in Mechanism:
            # 初始化结果列表
            results = []

            # 处理干净数据
            res_dict = testing_func(clean_df, clean_df, target, feature_schema, best_params)
            for algm in res_dict:
                clean_for_pg_mse = res_dict[algm][0]
                clean_for_pg_mae = res_dict[algm][1]
                results.append(["clean.csv", res_dict[algm][0], res_dict[algm][1], 0, 0])
                print("'clean.csv' is ok.")
                print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, 0, 0")

            # 处理脏数据
            # for rate in Missing_rate:
            #     input_dirty_file = os.path.join(input_base_path, dataset, "null", pattern, f'dirty-{rate}.csv')
            #     dirty_df = pd.read_csv(input_dirty_file).astype(str)
            #     dirty_df.fillna('nan', inplace=True)
            #     res_dict = testing_func(dirty_df, clean_df, target, feature_schema, best_params)
            #     for algm in res_dict:
            #         print(f"'dirty-{rate}.csv' is ok.")
            #         if res_dict[algm][0] < clean_for_pg_mse:
            #             dirty_for_pg_mse = 0
            #         else:
            #             dirty_for_pg_mse = (res_dict[algm][0] - clean_for_pg_mse) / clean_for_pg_mse
            #         if res_dict[algm][1] < clean_for_pg_mae:
            #             dirty_for_pg_mae = 0
            #         else:
            #             dirty_for_pg_mae = (res_dict[algm][1] - clean_for_pg_mae) / clean_for_pg_mae
            #         results.append(
            #             [f"dirty-{rate}.csv", res_dict[algm][0], res_dict[algm][1], dirty_for_pg_mse, dirty_for_pg_mae])
            #         print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {dirty_for_pg_mse}, {dirty_for_pg_mae}")

            # 处理插补数据
            for model in Imputation_Algorithms:
                for rate in Missing_rate:
                    imputed_path = os.path.join(input_base_path, dataset, "Imputation", pattern, f'null-{model}',
                                                f'dirty-{model}-{rate}.csv')
                    imputed_df = pd.read_csv(imputed_path).astype(str)
                    imputed_df.fillna('nan', inplace=True)
                    res_dict = testing_func(imputed_df, clean_df, target, feature_schema, best_params)
                    for algm in res_dict:
                        print(f"'dirty-{model}-{rate}.csv' is ok.")
                        if res_dict[algm][0] < clean_for_pg_mse:
                            dirty_for_pg_mse = 0
                        else:
                            dirty_for_pg_mse = (res_dict[algm][0] - clean_for_pg_mse) / clean_for_pg_mse
                        if res_dict[algm][1] < clean_for_pg_mae:
                            dirty_for_pg_mae = 0
                        else:
                            dirty_for_pg_mae = (res_dict[algm][1] - clean_for_pg_mae) / clean_for_pg_mae
                        results.append(
                            [f"dirty-{model}-{rate}.csv", res_dict[algm][0], res_dict[algm][1], dirty_for_pg_mse,
                             dirty_for_pg_mae])
                        print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {dirty_for_pg_mse}, {dirty_for_pg_mae}")

            output_results_file = os.path.join(output_base_path, "regression", dataset, pattern,
                                               f"cnn-imputation-results-{dataset}.csv")
            dir_path = os.path.dirname(output_results_file)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            results_df = pd.DataFrame(results, columns=["File Name", "MSE", "MAE", "PG(MSE)", "PG(MAE)"])
            results_df.to_csv(output_results_file, index=False)
            print(f"Results saved to {output_results_file}")

print("All tasks completed!")