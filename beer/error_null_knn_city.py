import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance

save_path='./null-city-knn/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def fill_city_column(file_path):
    df_copy = pd.read_csv(file_path)
    df = df_copy.replace("", pd.NA)
    missing = df[df['city'].isnull()]

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    df_drop = df.drop(columns=['city', 'id']).fillna(0)
    df_drop = df_drop.astype(float)
    df_drop_drop = df_drop.drop(missing.index)

    for index, row in missing.iterrows():
        df_missing_drop = df_drop.iloc[[index]]
        distances = [distance.euclidean(df_missing_drop.values[0], df_drop_drop.iloc[i].values) for i in range(len(df_drop_drop))]
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:5]
        nearest_cities = df.loc[df_drop_drop.index[nearest_indices], 'city'].values
        mode_city = pd.Series(nearest_cities).mode()[0]
        df.loc[index, 'city'] = mode_city

    for column in encoded_columns:
        df[column] = label_encoders[column].inverse_transform(df[column])

    df_copy['city'] = df['city']
    return df_copy

# 文件列表
file_names = ['dirty-city-10.csv', 'dirty-city-30.csv', 'dirty-city-50.csv', 'dirty-city-70.csv', 'dirty-city-90.csv']

# 处理每个文件并保存结果
for file_name in file_names:
    filled_df = fill_city_column(file_name)
    filled_file_name = file_name.replace('dirty-city', 'dirty-city-knn')
    filled_df.to_csv(os.path.join(save_path,filled_file_name), index=False)