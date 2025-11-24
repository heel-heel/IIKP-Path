import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras import layers, models
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def process_and_fill(input_file, output_file, target_column, unrelated_column):
    df = pd.read_csv(input_file)
    if 'quality' in df.columns:
        df = pd.read_csv(input_file, dtype={'quality': 'object'})
    elif 'Type' in df.columns:
        df = pd.read_csv(input_file, dtype={'Type': 'object'})
    else:
        df = pd.read_csv(input_file)
    df_copy = df.copy()

    if "Beers" in input_file:
        df['ounces'] = df['ounces'].fillna(df['ounces'].mean())
        df['abv'] = df['abv'].fillna(df['abv'].mean())
        df['ibu'] = df['ibu'].fillna(df['ibu'].mean())

    label_encoders = {}
    encoded_columns = []

    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_nan_values = df[column].dropna()
        df.loc[non_nan_values.index, column] = le.fit_transform(non_nan_values.astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    if "Beers" in input_file:
        df['style'] = df['style'].fillna(df['style'].mean())
        df['state'] = df['state'].fillna(df['state'].mean())

    if unrelated_column != "None":
        df_train = df.dropna(subset=[col for col in df.columns if col != target_column and col != unrelated_column])
    else:
        df_train = df.dropna(subset=[col for col in df.columns if col != target_column])

    df_train[target_column] = df_train[target_column].fillna(df_train[target_column].mean())

    scaler = MinMaxScaler()
    if unrelated_column != "None":
        X_scaled = scaler.fit_transform(df_train.drop(columns=[unrelated_column]))
    else:
        X_scaled = scaler.fit_transform(df_train)

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
    known_target_codes = df_train[target_column].dropna().values
    min_code = np.min(known_target_codes)
    max_code = np.max(known_target_codes)

    target_index = df.columns.get_loc(target_column)
    missing_indices = df_copy[df_copy[target_column].isnull()].index

    train_indices = df_train.index
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(train_indices)}

    for index in missing_indices:
        if index in index_mapping:
            generated_value = final_imputed_matrix[index_mapping[index], target_index]

            if generated_value < min_code:
                generated_value = int(min_code)
            elif generated_value > max_code:
                generated_value = int(max_code)
            else:
                distances = np.abs(known_target_codes - generated_value)
                closest_index = np.argmin(distances)
                generated_value = int(known_target_codes[closest_index])

            target_name = label_encoders[target_column].inverse_transform([int(generated_value)])[0]
            df_copy.at[index, target_column] = target_name

    df_copy.to_csv(output_file, index=False)
    print(f'{output_file} is been saved.')


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    unrelated_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, unrelated_column)