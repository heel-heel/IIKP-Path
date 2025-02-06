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

    # 步骤3: 处理“Score”列
    if 'Score' in df.columns:
        # 复制“Score”列
        df['Score_copy'] = df['Score'].copy()

        # 去除百分号并转换为数值类型，非数值的转换为NaN
        def clean_score(x):
            if isinstance(x, str) and x not in ['empty', '']:
                return x.strip().replace('%', '')
            return x

        df['Score_copy'] = df['Score_copy'].apply(clean_score)
        df['Score_copy'] = pd.to_numeric(df['Score_copy'], errors='coerce')

        # 计算非空的平均值
        mean_value = df['Score_copy'].dropna().mean()
        print(mean_value)

        # 如果平均值存在，则将其转换为字符串并添加百分号
        if not pd.isnull(mean_value):
            mean_value_str = f"{mean_value:.2f}%"  # 保留两位小数
        else:
            mean_value_str = ""


    # 将填充后的DataFrame导出到新的CSV文件
    if not df['Score'].empty:
        df['Score'].fillna(mean_value_str, inplace=True)

    df.to_csv(output_filename, index=False)

    print(f'File {output_filename} has been created.')