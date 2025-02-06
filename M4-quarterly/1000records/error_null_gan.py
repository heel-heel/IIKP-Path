import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

save_path='./null-gan/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def process_and_fill_csv(file_name, output_file_name):
    # 加载CSV文件
    df = pd.read_csv(file_name)

    # 假设缺失值用np.nan表示
    # 获取V2列中缺失值的索引
    missing_indices = df[df['V2'].isnull()].index

    # 删除缺失值行以训练GAN
    df_train = df.dropna(subset=['V2'])

    # 删除V1列
    df_train = df_train.drop(columns=['V1'])

    # 将数据转换为numpy数组
    data = df_train.values

    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

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
        generated_data = scaler.inverse_transform([generated_data])[0]  # 反标准化
        df.at[index, 'V2'] = int(np.round(generated_data[1]))  # 四舍五入

    # 将V1列加回来
    df['V1'] = df['V1'].fillna(method='ffill')  # 填充缺失的V1值

    # 导出填充后的CSV文件
    df.to_csv(os.path.join(save_path,output_file_name), index=False)

# 主程序
file_numbers = [10, 30, 50, 70, 90]
for number in file_numbers:
    input_file = f'dirty-{number}.csv'
    output_file = f'dirty-gan-{number}.csv'
    process_and_fill_csv(input_file, output_file)