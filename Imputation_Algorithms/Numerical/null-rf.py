import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor


def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    df = pd.read_csv(input_file)
    nonnumerical_data = df[nonnumerical_column].copy()
    df.drop(nonnumerical_column, axis=1, inplace=True)

    mask = df[target_column].isnull()
    df_complete = df[~mask]
    df_incomplete = df[mask]

    X_train = df_complete.drop(target_column, axis=1)
    y_train = df_complete[target_column]
    X_test = df_incomplete.drop(target_column, axis=1)
    rfc = RandomForestRegressor(n_estimators=100, random_state=0)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    df.loc[mask, target_column] = y_pred
    df = pd.concat([nonnumerical_data.to_frame(), df], axis=1)
    df.columns.values[0] = nonnumerical_column
    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)