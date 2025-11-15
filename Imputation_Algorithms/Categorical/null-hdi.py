import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance


def process_and_fill(input_file, output_file, target_column, unrelated_column):
    df_copy = pd.read_csv(input_file)
    df = df_copy.copy()
    missing = df[df[target_column].isnull()]

    label_encoders = {}
    encoded_columns = []
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
        encoded_columns.append(column)

    if unrelated_column != "None":
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
    if 'quality' in target_column:
        df_copy[target_column] = df_copy[target_column].astype(int)
    elif 'Type' in target_column:
        df_copy[target_column] = df_copy[target_column].astype(int)
    df_copy.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    unrelated_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, unrelated_column)