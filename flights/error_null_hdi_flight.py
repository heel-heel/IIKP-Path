import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance

save_path = './null-flight-hdi/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

read_path = './null-flight/'

def process_and_fill_csv(file_name, output_file_name):
    df_copy = pd.read_csv(os.path.join(read_path,file_name))
    df = df_copy.replace("", pd.NA)
    missing = df[df['flight'].isnull()]

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    df_drop = df.drop(columns=['flight']).fillna(0)
    df_drop = df_drop.astype(float)
    df_drop_drop = df_drop.drop(missing.index)

    for index, row in missing.iterrows():
        df_missing_drop = df_drop.iloc[[index]]
        distances = [distance.euclidean(df_missing_drop.values[0], df_drop_drop.iloc[i].values) for i in
                     range(len(df_drop_drop))]
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:1]
        nearest_cities = df.loc[df_drop_drop.index[nearest_indices], 'flight'].values
        mode_city = pd.Series(nearest_cities).mode()[0]
        df.loc[index, 'flight'] = mode_city

    for column in encoded_columns:
        df[column] = label_encoders[column].inverse_transform(df[column])

    df_copy['flight'] = df['flight']
    df_copy.to_csv(os.path.join(save_path, output_file_name), index=False)


Missing_rate = [10, 30, 50, 70, 90]
for rate in Missing_rate:
    input_file = f'dirty-flight-{rate}.csv'
    output_file = f'dirty-flight-hdi-{rate}.csv'
    process_and_fill_csv(input_file, output_file)