import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    df_copy = pd.read_csv(input_file)
    df = df_copy.drop(columns=[nonnumerical_column])
    df[target_column] = df[target_column].fillna(df[target_column].mean())

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df)

    noise_factor = 0.5
    X_noisy = X_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_scaled.shape)
    X_noisy = np.clip(X_noisy, 0.0, 1.0)
    input_dim = X_scaled.shape[1]
    encoding_dim = 5

    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(X_noisy, X_scaled, epochs=100, batch_size=10, shuffle=True,
                validation_split=0.2, callbacks=[early_stopping])

    c = 10
    imputed_matrices = []
    for _ in range(c):
        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(X_noisy, X_scaled, epochs=100, batch_size=10, shuffle=True,
                    validation_split=0.2, callbacks=[early_stopping])
        imputed_matrix = autoencoder.predict(X_scaled)
        imputed_matrices.append(imputed_matrix)
    final_imputed_matrix = np.mean(imputed_matrices, axis=0)
    final_imputed_matrix = scaler.inverse_transform(final_imputed_matrix)
    target_index = df.columns.get_loc(target_column)
    df_copy[target_column] = np.where(df_copy[target_column].isnull(), final_imputed_matrix[:, target_index], df_copy[target_column])

    df_copy.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')



datasets = {
    "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"}
}
Missing_rate = [10, 30, 50, 70, 90]
base_path = "../../Datasets"
for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]
    input_path = os.path.join(base_path, dataset, "null")
    output_path = os.path.join(base_path, dataset, "Imputation", "null-midae")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in Missing_rate:
        input_file = os.path.join(input_path, f'dirty-{rate}.csv')
        output_file = os.path.join(output_path, f'dirty-midae-{rate}')
        process_and_fill(input_file, output_file, target_column, nonnumerical_column)