import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 设置保存路径
save_path = './null-city-midae/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

read_path = './null-city/'


def process_and_fill_csv(file_name, output_file_name):
    df_copy = pd.read_csv(os.path.join(read_path, file_name))
    df = df_copy.copy()

    # 填充数值列的缺失值
    df['ounces'] = df['ounces'].fillna(df['ounces'].mean())
    df['abv'] = df['abv'].fillna(df['abv'].mean())
    df['ibu'] = df['ibu'].fillna(df['ibu'].mean())

    # 处理非数值属性（city列）
    label_encoders = {}
    encoded_columns = []

    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        non_nan_values = df[column].dropna()
        df.loc[non_nan_values.index, column] = le.fit_transform(non_nan_values.astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    # 删除除city列以外其他列有缺失值的行
    df_train = df.dropna(subset=[col for col in df.columns if col != 'city' and col != 'id'])

    # 均值填充city列缺失值，以便训练DAE模型
    df_train['city'] = df_train['city'].fillna(df_train['city'].mean())

    # 归一化数据
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_train.drop(columns=['id']))

    # 添加噪声
    noise_factor = 0.5
    X_noisy = X_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_scaled.shape)
    X_noisy = np.clip(X_noisy, 0.0, 1.0)

    # 构建去噪自编码器
    input_dim = X_scaled.shape[1]
    encoding_dim = 5  # 编码维度

    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # 设置早停法回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练自编码器
    autoencoder.fit(X_noisy, X_scaled, epochs=100, batch_size=10, shuffle=True,
                    validation_split=0.2, callbacks=[early_stopping])

    # 多次运行DAE模型
    c = 10  # 运行次数
    imputed_matrices = []

    for _ in range(c):
        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(X_noisy, X_scaled, epochs=100, batch_size=10, shuffle=True,
                        validation_split=0.2, callbacks=[early_stopping])
        imputed_matrix = autoencoder.predict(X_scaled)
        imputed_matrices.append(imputed_matrix)

    # 平均多次运行的结果
    final_imputed_matrix = np.mean(imputed_matrices, axis=0)

    # 反归一化
    final_imputed_matrix = scaler.inverse_transform(final_imputed_matrix)

    # 获取已知的城市编码范围
    known_city_codes = df_train['city'].dropna().values
    min_code = np.min(known_city_codes)
    max_code = np.max(known_city_codes)

    # 填充city列的缺失值，但不改变非空缺位置的数值
    city_index = df.columns.get_loc('city')
    missing_indices = df_copy[df_copy['city'].isnull()].index

    # 调整索引映射
    train_indices = df_train.index
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(train_indices)}

    for index in missing_indices:
        if index in index_mapping:
            generated_value = final_imputed_matrix[index_mapping[index], city_index]

            # 确保生成的数值在已知的城市编码范围内
            if generated_value < min_code:
                generated_value = int(min_code)
            elif generated_value > max_code:
                generated_value = int(max_code)
            else:
                distances = np.abs(known_city_codes - generated_value)
                closest_index = np.argmin(distances)
                generated_value = int(known_city_codes[closest_index])

            city_name = label_encoders['city'].inverse_transform([int(generated_value)])[0]
            df_copy.at[index, 'city'] = city_name

    # 导出填充后的CSV文件
    df_copy.to_csv(os.path.join(save_path, output_file_name), index=False)
    print(f"填充后的数据已导出到 {output_file_name}")


# 处理不同缺失率的文件
Missing_rate = [10,30,50,70,90]
for rate in Missing_rate:
    input_file = f'dirty-city-{rate}.csv'
    output_file = f'dirty-city-midae-{rate}.csv'
    process_and_fill_csv(input_file, output_file)