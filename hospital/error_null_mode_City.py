import pandas as pd
import os

save_path='./null-City-mode/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

read_path = './null-City/'

# 缺失率列表
missing_rates = ['10', '30', '50', '70', '90']

# 遍历每个缺失率
for rate in missing_rates:
    # 构建输入和输出文件名
    input_filename = f'dirty-{rate}.csv'
    output_filename = f'dirty-City-mode-{rate}.csv'

    # 步骤2: 读取CSV文件
    df = pd.read_csv(os.path.join(read_path,input_filename))

    # 步骤3: 找到“city”列的众数
    # 确保“city”列不为空再找众数
    if not df['City'].empty:
        city_mode = df['City'].mode()[0]  # mode()返回的是一个Series，取第一个值作为众数

        # 步骤4: 用众数填充“city”列中的缺失值
        df['City'].fillna(city_mode, inplace=True)

    # 步骤5: 将填充后的DataFrame导出到新的CSV文件
    df.to_csv(os.path.join(save_path,output_filename), index=False)

    print(f'File {output_filename} has been created.')