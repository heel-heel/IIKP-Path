import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 设置保存路径
save_path = './null-midae/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

read_path = './null/'

def process_and_fill_csv(file_name, output_file_name):
    df_copy = pd.read_csv(os.path.join(read_path, file_name))
    df = df_copy.drop(columns=['V1'])
    df['V2'] = df['V2'].fillna(df['V2'].mean())

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
    v2_index = df.columns.get_loc('V2')
    df_copy['V2'] = np.where(df_copy['V2'].isnull(), final_imputed_matrix[:, v2_index], df_copy['V2'])

    df_copy.to_csv(os.path.join(save_path, output_file_name), index=False)
    print(f"填充后的数据已导出到 {output_file}")

Missing_rate = [10,30,50,70,90]
for rate in Missing_rate:
    input_file = f'dirty-{rate}.csv'
    output_file = f'dirty-midae-{rate}.csv'
    process_and_fill_csv(input_file, output_file)