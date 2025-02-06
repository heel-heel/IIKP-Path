import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance

save_path='./null-City-knn/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def fill_city_column(file_path):
    # 导入CSV文件
    df_copy = pd.read_csv(file_path)

    # 将空字符串替换为NaN，以便后续处理
    df = df_copy.replace("", pd.NA)

    # 获取没有缺失值的记录
    non_missing = df[df['City'].notnull()]
    # 获取缺失值的记录
    missing = df[df['City'].isnull()]

    # 对所有非数值属性列进行编码
    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    df_drop = df.drop(columns=['City','ProviderNumber']).fillna(0)
    df_drop = df_drop.astype(float)
    df_drop_drop = df_drop.drop(missing.index)

    # 对于missing中的每一条记录，找到在df_drop中距离最近的5条记录
    for index, row in missing.iterrows():
        # 取出对应的记录，形成df_missing_drop
        df_missing_drop = df_drop.iloc[[index]]

        # 计算df_missing_drop与df_drop中每条记录的欧几里得距离
        distances = [distance.euclidean(df_missing_drop.values[0], df_drop_drop.iloc[i].values) for i in range(len(df_drop_drop))]

        # 根据距离排序，找到最近的5条记录的索引
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:5]

        # 使用这些索引来获取最接近的5条记录的city值
        nearest_cities = df.loc[df_drop_drop.index[nearest_indices], 'City'].values

        # 计算这5个city值的众数
        mode_City = pd.Series(nearest_cities).mode()[0]

        # 将众数赋值给df中index对应的记录的city列
        df.loc[index, 'City'] = mode_City

    # 将编码后的非数值属性列转换回原始字符串类型
    for column in encoded_columns:
        df[column] = label_encoders[column].inverse_transform(df[column])

    # 将df的city列赋值给df_copy的city列
    df_copy['City'] = df['City']

    # 保存更新后的DataFrame
    return df_copy

# 文件列表
file_names = ['dirty-City-10.csv', 'dirty-City-30.csv', 'dirty-City-50.csv', 'dirty-City-70.csv', 'dirty-City-90.csv']

# 处理每个文件并保存结果
for file_name in file_names:
    filled_df = fill_city_column(file_name)
    filled_file_name = file_name.replace('dirty-City', 'dirty-City-knn')
    filled_df.to_csv(os.path.join(save_path,filled_file_name), index=False)