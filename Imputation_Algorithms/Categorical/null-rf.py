import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def process_and_fill(input_file, output_file, target_column, unrelated_column):
    df = pd.read_csv(input_file)
    df_copy = df.copy()
    mask = df[target_column].isnull()

    # 创建标签编码器
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

    # 构建训练集和测试集
    X_train = df_complete.drop(target_column, axis=1)
    y_train = df_complete[target_column]
    X_test = df_incomplete.drop(target_column, axis=1)

    # 训练随机森林模型
    rfc = RandomForestRegressor(n_estimators=100, random_state=0)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    y_pred_rounded = [round(value) for value in y_pred]
    df_drop.loc[df_incomplete.index, target_column] = y_pred_rounded
    for column in df_drop.columns:
        df_drop[column] = label_encoders[column].inverse_transform(df_drop[column])

    df_copy[target_column]=df_drop[target_column]
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
    output_path = os.path.join(base_path, dataset, "Imputation", "null-rf")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for rate in Missing_rate:
        input_file = os.path.join(input_path, f'dirty-{rate}.csv')
        output_file = os.path.join(output_path, f'dirty-rf-{rate}')
        process_and_fill(input_file, output_file, target_column, unrelated_column)