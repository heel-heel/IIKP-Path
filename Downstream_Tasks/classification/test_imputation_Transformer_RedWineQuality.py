import os
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

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

# Define the parameter grid for random search
#param_dist = {
#    'd_model': [32, 64, 128, 256],
#    'nhead': [2, 4, 8],
#    'num_layers': [1, 2, 3, 4],
#    'dim_feedforward': [64, 128, 256, 512],
#    'dropout': [0.1, 0.2, 0.3, 0.4],
#    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
#    'batch_size': [16, 32, 64, 128],
#    'epochs': [50, 100, 200, 300]
#}

param_dist = {
    'd_model': [128],
    'nhead': [2],
    'num_layers': [8],
    'dim_feedforward': [64],
    'dropout': [0.1],
    'learning_rate': [0.0001],
    'batch_size': [16],
    'epochs': [10]
}

datasets = {
    #"Beers": {"target_column": "city", "unrelated_column": "id"},
    #"Flights": {"target_column": "flight", "unrelated_column": None},
    #"Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"},
    "RedWineQuality": {"target_column": "quality", "unrelated_column": "None"},
    #"AvocadoRipeness": {"target_column": "ripeness", "unrelated_column": "None"}
}
Imputation_Algorithms = ['mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
Mechanism = ["MCAR", "MAR", "MNAR"]

class TabularTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super(TabularTransformer, self).__init__()

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding (optional, for sequence-like processing)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # Project input
        x = self.input_projection(x)

        # Add positional encoding (reshape for transformer)
        x = x.unsqueeze(1)  # Add sequence dimension: (batch, seq_len=1, d_model)
        x = x + self.pos_encoding

        # Pass through transformer
        x = self.transformer_encoder(x)

        # Remove sequence dimension
        x = x.squeeze(1)

        # Layer normalization
        x = self.layer_norm(x)

        # Output
        x = self.output_layer(x)
        return x


# Scikit-learn compatible wrapper for Transformer
class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                 dropout=0.1, learning_rate=0.001, batch_size=32, epochs=100,
                 random_state=42, device='cpu'):
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
        self.optimizer = None
        self.criterion = None
        self.label_encoder = None
        self.scaler = StandardScaler()

    def _encode_labels(self, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            return self.label_encoder.fit_transform(y)
        return self.label_encoder.transform(y)

    def _decode_labels(self, y):
        return self.label_encoder.inverse_transform(y)

    def fit(self, X, y):
        # Set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Encode labels
        y_encoded = self._encode_labels(y)
        num_classes = len(self.label_encoder.classes_)

        self.classes_ = self.label_encoder.classes_

        # Scale features
        X_scaled = self.scaler.fit_transform(X.astype(np.float32))

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        self.model = TabularTransformer(
            input_dim=X.shape[1],
            num_classes=num_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)

        # Initialize optimizer and criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {total_loss / len(dataloader):.4f}")

        return self

    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X.astype(np.float32))
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()

        return self._decode_labels(predicted)

    def predict_proba(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X.astype(np.float32))
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

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


def transformer_model(X_train, X_test, y_train, y_test, best_params=None):
    if best_params:
        model = TransformerClassifier(**best_params, random_state=42)
    else:
        model = TransformerClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1 = evaluate(y_test, y_pred)
    return precision, recall, f1


def testing_func(rep_df, clean_df, target, feature_schema, best_params=None):
    feature_schema = [x.lower() for x in feature_schema]
    rep_df.columns = rep_df.columns.str.lower()
    clean_df.columns = clean_df.columns.str.lower()
    target = target.lower()
    df_encoded = pd.get_dummies(rep_df[feature_schema])

    sanitized_feature_names = {}
    for feature_names_str in df_encoded.columns:
        valid_chars = [char for char in feature_names_str if char not in ['[', ']', '<']]
        sanitized_feature_names[feature_names_str] = ''.join(valid_chars)

    df_encoded.rename(columns=sanitized_feature_names, inplace=True)
    train_indices, test_indices = train_test_split(range(len(df_encoded)), test_size=0.2, random_state=42)
    X_train = df_encoded.iloc[train_indices].values.astype(np.float32)
    y_train = rep_df[target].iloc[train_indices].values
    X_test = df_encoded.iloc[test_indices].values.astype(np.float32)
    y_test = clean_df[target].iloc[test_indices].values

    res_dict = {}
    pre, rec, f1 = transformer_model(X_train, X_test, y_train, y_test, best_params)
    res_dict['transformer'] = [pre, rec, f1]
    return res_dict


def evaluate(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return precision, recall, f1


def perform_random_search(X_train, y_train):
    transformer_clf = TransformerClassifier(random_state=42)

    # Create a wrapper for randomized search with reduced iterations
    param_dist_limited = {k: v for k, v in param_dist.items() if
                          k in ['d_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout', 'learning_rate',
                                'batch_size', 'epochs']}

    random_search = RandomizedSearchCV(
        transformer_clf,
        param_distributions=param_dist_limited,
        #cv=5,  # Reduced CV folds for transformer
        cv=2,
        random_state=42,
        n_jobs=-1,
        verbose=10,
        #n_iter=50,  # Reduced iterations due to longer training time
        n_iter=1,
        scoring=make_scorer(f1_score, average='weighted', zero_division=0)
    )
    random_search.fit(X_train, y_train)
    return random_search.best_params_


if __name__ == "__main__":
    input_base_path = "../../Datasets"
    output_base_path = "../../Downstream_Results"

    for dataset, columns in datasets.items():
        print('-' * 70)
        print(f"{dataset}:Processing...")
        target = columns["target_column"]
        clean_path = os.path.join(input_base_path, dataset, 'clean.csv')
        clean_df = pd.read_csv(clean_path).astype(str)
        if 'quality' in clean_df.columns:
            clean_df = pd.read_csv(clean_path, dtype={'quality': 'object'}).astype(str)
        else:
            clean_df = pd.read_csv(clean_path).astype(str)
        clean_df.fillna('nan', inplace=True)
        feature_schema = list(clean_df.columns)
        feature_schema.remove(target)

        # Prepare data for random search
        clean_df_encoded = pd.get_dummies(clean_df[feature_schema])
        sanitized_feature_names = {}
        for feature_names_str in clean_df_encoded.columns:
            valid_chars = [char for char in feature_names_str if char not in ['[', ']', '<']]
            sanitized_feature_names[feature_names_str] = ''.join(valid_chars)
        clean_df_encoded.rename(columns=sanitized_feature_names, inplace=True)

        train_indices, test_indices = train_test_split(range(len(clean_df_encoded)), test_size=0.2, random_state=42)
        X_train_search = clean_df_encoded.iloc[train_indices].values.astype(np.float32)
        y_train_search = clean_df[target].iloc[train_indices].values

        # Perform random search only on clean.csv
        print("Performing random search for best parameters...")
        best_params = perform_random_search(X_train_search, y_train_search)
        print(f"Best parameters found: {best_params}")

        for pattern in Mechanism:
            # Initialize results list
            results = []

            # Process clean data with best parameters
            res_dict = testing_func(clean_df, clean_df, target, feature_schema, best_params)
            for algm in res_dict:
                clean_for_pg = res_dict[algm][2]
                results.append(["clean.csv", res_dict[algm][0], res_dict[algm][1], res_dict[algm][2], 0])
                print("'clean.csv' is ok.")
                print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}, 0")

            # Process dirty data with best parameters
            for rate in Missing_rate:
                input_dirty_file = os.path.join(input_base_path, dataset, "null", pattern, f'dirty-{rate}.csv')
                dirty_df = pd.read_csv(input_dirty_file).astype(str)
                if 'quality' in dirty_df.columns:
                    dirty_df = pd.read_csv(input_dirty_file, dtype={'quality': 'object'}).astype(str)
                else:
                    dirty_df = pd.read_csv(input_dirty_file).astype(str)
                dirty_df.fillna('nan', inplace=True)
                res_dict = testing_func(dirty_df, clean_df, target, feature_schema, best_params)
                for algm in res_dict:
                    print(f"'dirty-{rate}.csv' is ok.")
                    if res_dict[algm][2] > clean_for_pg:
                        results.append(
                            [f'dirty-{rate}.csv', res_dict[algm][0], res_dict[algm][1], res_dict[algm][2], 0])
                        print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}, 0")
                    else:
                        results.append([f'dirty-{rate}.csv', res_dict[algm][0], res_dict[algm][1], res_dict[algm][2],
                                        (clean_for_pg - res_dict[algm][2]) / clean_for_pg])
                        print(
                            f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}, {(clean_for_pg - res_dict[algm][2]) / clean_for_pg}")

            # Process imputed data with best parameters
            for model in Imputation_Algorithms:
                for rate in Missing_rate:
                    imputed_path = os.path.join(input_base_path, dataset, "Imputation", pattern, f'null-{model}',
                                                f'dirty-{model}-{rate}.csv')
                    imputed_df = pd.read_csv(imputed_path).astype(str)
                    if 'quality' in imputed_df.columns:
                        imputed_df = pd.read_csv(imputed_path, dtype={'quality': 'object'}).astype(str)
                    else:
                        imputed_df = pd.read_csv(imputed_path).astype(str)
                    imputed_df.fillna('nan', inplace=True)
                    res_dict = testing_func(imputed_df, clean_df, target, feature_schema, best_params)
                    for algm in res_dict:
                        print(f"'dirty-{model}-{rate}.csv' is ok.")
                        if res_dict[algm][2] > clean_for_pg:
                            results.append(
                                [f'dirty-{model}-{rate}.csv', res_dict[algm][0], res_dict[algm][1], res_dict[algm][2],
                                 0])
                            print(f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}, 0")
                        else:
                            results.append(
                                [f'dirty-{model}-{rate}.csv', res_dict[algm][0], res_dict[algm][1], res_dict[algm][2],
                                 (clean_for_pg - res_dict[algm][2]) / clean_for_pg])
                            print(
                                f"{res_dict[algm][0]}, {res_dict[algm][1]}, {res_dict[algm][2]}, {(clean_for_pg - res_dict[algm][2]) / clean_for_pg}")

            output_results_file = os.path.join(output_base_path, "classification", dataset, pattern,
                                               f"transformer-imputation-results-{dataset}.csv")
            dir_path = os.path.dirname(output_results_file)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            results_df = pd.DataFrame(results, columns=["File Name", "Precision", "Recall", "F1 Score", "PG(F1 Score)"])
            results_df.to_csv(output_results_file, index=False)

print("All tasks completed!")