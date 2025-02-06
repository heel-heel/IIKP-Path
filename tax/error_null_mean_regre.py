import pandas as pd

# 缺失率列表
missing_rates = ['10', '30', '50', '70', '90']

# 遍历每个缺失率
for rate in missing_rates:
    # 构建输入和输出文件名
    input_filename = f'dirty-{rate}.csv'
    output_filename = f'dirty-mean-{rate}.csv'

    # 步骤2: 读取CSV文件
    df = pd.read_csv(input_filename)

    # 步骤3: 找到“ounces”列的平均数
    # 确保“ounces”列不为空再求平均数
    if not df['rate'].empty:
        city_mode = df['rate'].mean()  # 平均数

        # 步骤4: 用平均数填充“ounces”列中的缺失值
        df['rate'].fillna(city_mode, inplace=True)

    # 步骤5: 将填充后的DataFrame导出到新的CSV文件
    df.to_csv(output_filename, index=False)

    print(f'File {output_filename} has been created.')