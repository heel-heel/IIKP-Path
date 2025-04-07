import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def process_and_fill(input_file, output_file, target_column, unrelated_column):
    df = pd.read_csv(input_file)
    df_copy = df.copy()
    mask = df[target_column].isnull()

    label_encoders = {}
    for column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    if unrelated_column is not None:
        df_drop = df.drop(columns=[unrelated_column])
    else:
        df_drop = df
    df_complete = df_drop[~mask]
    df_incomplete = df_drop[mask]

    X_train = df_complete.drop(target_column, axis=1)
    y_train = df_complete[target_column]
    X_test = df_incomplete.drop(target_column, axis=1)

    previous_imputed_values = None
    average_difference = 0
    iteration = 0
    max_iterations = 100

    while iteration < max_iterations:
        rfc = RandomForestRegressor(n_estimators=100, random_state=42)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)

        if previous_imputed_values is not None:
            current_difference = np.mean(np.abs(y_pred - previous_imputed_values))
            if current_difference > average_difference:
                print("Stopping criterion met: average difference increased.")
                break
            average_difference = current_difference

        previous_imputed_values = y_pred
        df.loc[mask, target_column] = y_pred
        iteration += 1



if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    unrelated_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, unrelated_column)