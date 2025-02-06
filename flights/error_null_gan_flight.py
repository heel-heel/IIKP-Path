import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

save_path='./null-flight-gan/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

read_path = './null-flight/'

def process_and_fill_csv(file_name, output_file_name):
    df_copy = pd.read_csv(os.path.join(read_path,file_name))
    df = df_copy.replace("", pd.NA)
    missing_indices = df[df['flight'].isnull()].index

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    # 删除缺失值行以训练GAN
    df_train = df.fillna(0)
    df_train = df_train.astype(float)
    df_train = df_train.drop(missing_indices)
    #print(df_train['flight'])

    data_scaled = df_train.values

    # 定义GAN的生成器和判别器
    def make_generator_model(data_shape):
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=(data_shape,)))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(data_shape))
        return model

    def make_discriminator_model(data_shape):
        model = tf.keras.Sequential()
        model.add(layers.Dense(256, activation='relu', input_shape=(data_shape,)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    # 创建GAN模型
    generator = make_generator_model(data_scaled.shape[1])
    discriminator = make_discriminator_model(data_scaled.shape[1])

    # 定义GAN的训练过程
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(data):
        noise = tf.random.normal([data.shape[0], data.shape[1]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = generator(noise, training=True)

            real_output = discriminator(data, training=True)
            fake_output = discriminator(generated_data, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(dataset, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for data in dataset:
                gen_loss, disc_loss = train_step(data)
                # 打印损失值
                print(f"Generator Loss: {gen_loss.numpy():.4f}, Discriminator Loss: {disc_loss.numpy():.4f}")

    # 准备数据集
    batch_size = 25
    dataset = tf.data.Dataset.from_tensor_slices(data_scaled).shuffle(len(data_scaled)).batch(batch_size)

    # 训练GAN
    epochs = 2000
    train(dataset, epochs)

    # 使用生成器填充缺失值
    for index in missing_indices:
        noise = tf.random.normal([1, data_scaled.shape[1]])
        generated_data = generator(noise, training=False)
        generated_data = generated_data.numpy().flatten()

        # 获取生成的数值
        generated_value = generated_data[1]

        # 确保生成的数值在已知的城市编码范围内
        known_city_codes = df_train['flight'].values
        min_code = np.min(known_city_codes)
        max_code = np.max(known_city_codes)

        # 如果生成的数值超出范围，选择最接近的有效编码
        if generated_value < min_code:
            generated_value = min_code
        elif generated_value > max_code:
            generated_value = max_code
        else:
            distances = np.abs(known_city_codes - generated_value)
            closest_index = np.argmin(distances)
            generated_value = int(known_city_codes[closest_index])
        city_name = label_encoders['flight'].inverse_transform([int(generated_value)])[0]
        df.at[index, 'flight'] = city_name
        df_copy.at[index, 'flight'] = df.at[index, 'flight']
    df_copy.to_csv(os.path.join(save_path,output_file_name), index=False)

# 主程序
file_numbers = [10,30,50,70,90]
for number in file_numbers:
    input_file = f'dirty-flight-{number}.csv'
    output_file = f'dirty-flight-gan-{number}.csv'
    process_and_fill_csv(input_file, output_file)