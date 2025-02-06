import pandas as pd
from sklearn.impute import KNNImputer

# 读取CSV文件
df = pd.read_csv('dirty-10.csv')

# 确保日期列是日期类型，如果不是，可以将其转换为日期类型
df['date'] = pd.to_datetime(df['date'])

# 创建KNNImputer实例，这里假设我们使用5个邻居
imputer = KNNImputer(n_neighbors=5)

# 选择需要填充的列，这里我们假设只有'Ozone'列有缺失值
imputed_data = imputer.fit_transform(df[['Ozone']])

# 将填充后的数据转换回DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=['Ozone'], index=df.index)

# 将填充后的DataFrame合并回原始DataFrame
df['Ozone'] = imputed_df['Ozone']

# 检查填充结果
df.to_csv('pollution_imputed.csv', index=False)