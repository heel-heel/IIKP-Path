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


datasets = {
    "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"}
}
Missing_rate = [10, 30, 50, 70, 90]
base_path = "../../Datasets"
for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]
    input_path = os.path.join(base_path, dataset, "null")
    output_path = os.path.join(base_path, dataset, "Imputation", "null-rf")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in Missing_rate:
        input_file = os.path.join(input_path, f'dirty-{rate}.csv')
        output_file = os.path.join(output_path, f'dirty-rf-{rate}.csv')
        process_and_fill(input_file, output_file, target_column, nonnumerical_column)