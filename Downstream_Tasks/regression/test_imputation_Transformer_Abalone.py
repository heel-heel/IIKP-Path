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

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x.squeeze()


# Scikit-learn compatible wrapper for Transformer Regressor
class TransformerRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=None, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=256, dropout=0.1, learning_rate=0.001,
                 batch_size=64, epochs=200, random_state=42, device='cpu'):
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.scaler = StandardScaler()

    def _create_model(self, input_dim):
        return TransformerRegressor(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
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
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
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


def get_best_transformer_params_with_cv(X_train, y_train, n_folds=5, n_iter=20):
    """使用RandomizedSearchCV进行参数搜索"""

    # 参数分布
    #param_dist = {
    #    'd_model': [64, 128, 256],
    #    'nhead': [4, 8, 16],
    #    'num_layers': [2, 3, 4],
    #    'dim_feedforward': [128, 256, 512],
    #    'dropout': [0.1, 0.2, 0.3],
    #    'learning_rate': [0.001, 0.0005, 0.0001],
    #    'batch_size': [32, 64, 128],
    #    'epochs': [100, 200, 300]
    #}

    param_dist = {
        'd_model': [64],
        'nhead': [8],
        'num_layers': [3],
        'dim_feedforward': [128],
        'dropout': [0.1],
        'learning_rate': [0.1],
        'batch_size': [32],
        'epochs': [1]
     }

    best_params = {
        'd_model': random.choice(param_dist['d_model']),
        'nhead': random.choice(param_dist['nhead']),
        'num_layers': random.choice(param_dist['num_layers']),
        'dim_feedforward': random.choice(param_dist['dim_feedforward']),
        'dropout': random.choice(param_dist['dropout']),
        'learning_rate': random.choice(param_dist['learning_rate']),
        'batch_size': random.choice(param_dist['batch_size']),
        'epochs': random.choice(param_dist['epochs']),
    }

    # # 创建包装器模型
    # transformer_reg = TransformerRegressorWrapper(input_dim=X_train.shape[1], random_state=42)
    #
    # # 创建评分器（使用负MAE，因为sklearn期望最大化分数）
    # scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    #
    # # 创建RandomizedSearchCV对象
    # random_search = RandomizedSearchCV(
    #     transformer_reg,
    #     param_distributions=param_dist,
    #     n_iter=n_iter,
    #     cv=n_folds,
    #     scoring=scorer,
    #     n_jobs=1,  # PyTorch模型建议使用1
    #     verbose=10,
    #     random_state=42
    # )
    #
    # # 执行搜索
    # random_search.fit(X_train, y_train)
    #
    # return random_search.best_params_
    return best_params


def transformer_regressor(X_train, X_test, y_train, y_test, best_params=None):
    """使用Transformer进行回归"""
    # 创建模型
    if best_params is None:
        model = TransformerRegressorWrapper(input_dim=X_train.shape[1], random_state=42)
    else:
        model = TransformerRegressorWrapper(
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
    mse, mae = transformer_regressor(X_train, X_test, y_train, y_test, best_params)
    res_dict['transformer'] = [mse, mae]
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
        #"concrete": {"target_column": "concrete_compressive_strength", "nonnumerical_column": "None"},
        #"CCPP": {"target_column": "PE", "nonnumerical_column": "None"},
        #"AirfoilSelfNoise": {"target_column": "SSPL", "nonnumerical_column": "None"},
        "Abalone": {"target_column": "Rings", "nonnumerical_column": "None"},
        #"ParisHousing": {"target_column": "price", "nonnumerical_column": "None"},
    }

    Imputation_Algorithms = [#'mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim',
                             'si',
                             #'mfi', 'missfi', 'xgbi',
                             #'gain', 'midae'
                             ]
    Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    Mechanism = ["MCAR", "MAR", "MNAR"]

    for dataset, columns in datasets.items():
        print('-' * 70)
        print(f"{dataset}:Processing...")
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

        #best_params = get_best_transformer_params_with_cv(X_train, y_train, n_folds=5, n_iter=50)
        best_params = get_best_transformer_params_with_cv(X_train, y_train, n_folds=2, n_iter=1)
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
                                               f"transformer-imputation-results-{dataset}.csv")
            dir_path = os.path.dirname(output_results_file)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            results_df = pd.DataFrame(results, columns=["File Name", "MSE", "MAE", "PG(MSE)", "PG(MAE)"])
            results_df.to_csv(output_results_file, index=False)