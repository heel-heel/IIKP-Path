import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance


def process_and_fill(input_file, output_file, target_column, unrelated_column):
    df_copy = pd.read_csv(input_file)
    df = df_copy
    missing = df[df[target_column].isnull()]

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    if unrelated_column is not None:
        df_drop = df.drop(columns=[target_column, unrelated_column]).fillna(0)
    else:
        df_drop = df.drop(columns=[target_column]).fillna(0)

    df_drop = df_drop.astype(float)
    df_drop_drop = df_drop.drop(missing.index)

    for index, row in missing.iterrows():
        df_missing_drop = df_drop.iloc[[index]]
        distances = [distance.euclidean(df_missing_drop.values[0], df_drop_drop.iloc[i].values) for i in
                     range(len(df_drop_drop))]
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:1]
        nearest_cities = df.loc[df_drop_drop.index[nearest_indices], target_column].values
        mode_target = pd.Series(nearest_cities).mode()[0]
        df.loc[index, target_column] = mode_target

    for column in encoded_columns:
        df[column] = label_encoders[column].inverse_transform(df[column])

    df_copy[target_column] = df[target_column]
    df_copy.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


datasets = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    "Flights": {"target_column": "flight", "unrelated_column": None},
    "Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"}
}
Missing_rate = [10, 30, 50, 70, 90]
base_path = "../../Datasets"
for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    unrelated_column = columns["unrelated_column"]
    input_path = os.path.join(base_path, dataset, "null")
    output_path = os.path.join(base_path, dataset, "Imputation", "null-hdi")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in Missing_rate:
        input_file = os.path.join(input_path, f'dirty-{rate}.csv')
        output_file = os.path.join(output_path, f'dirty-hdi-{rate}')
        process_and_fill(input_file, output_file, target_column, unrelated_column)