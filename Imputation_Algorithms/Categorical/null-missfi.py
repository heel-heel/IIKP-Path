import pandas as pd
import os
import sys
import numpy as np
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

    if unrelated_column != "None":
        df_drop = df.drop(columns=[unrelated_column])
    else:
        df_drop = df
    df_complete = df_drop[~mask]
    df_incomplete = df_drop[mask]

    X_train_initial = df_complete.drop(target_column, axis=1)
    y_train_initial = df_complete[target_column]
    X_test_initial = df_incomplete.drop(target_column, axis=1)

    rfc_initial = RandomForestRegressor(n_estimators=100, random_state=42)
    rfc_initial.fit(X_train_initial, y_train_initial)
    y_pred_initial = rfc_initial.predict(X_test_initial)

    # 显式地将 y_pred_initial 转换为与目标列相同的数据类型
    y_pred_initial = np.round(y_pred_initial).astype(int)
    df_drop.loc[mask, target_column] = y_pred_initial

    previous_imputed_values = df[target_column].copy()
    average_difference = 0
    iteration = 0
    max_iterations = 1

#修改，因为只在一个属性上有缺失值，迭代过程没有意义，所以改为在每个元组之间的迭代
    while iteration < max_iterations:
        current_imputed_values = df_drop[target_column].copy()
        for index, row in df_incomplete.iterrows():
            print(f"正在处理{index}...")
            df_temp = df_drop.drop(index)
            X_train = df_temp.drop(target_column, axis=1)
            y_train = df_temp[target_column]
            X_test = pd.DataFrame([row.drop(target_column)], columns=X_train.columns)

            rfc = RandomForestRegressor(n_estimators=100, random_state=42)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            y_pred_rounded = [round(value) for value in y_pred]
            df_drop.loc[index, target_column] = y_pred_rounded

        current_difference = np.mean(np.abs(df_drop[target_column] - previous_imputed_values))
        print(f"Iteration {iteration + 1}: {current_difference},{average_difference}")

        if iteration > 1 and current_difference >= average_difference:
            print("Stopping criterion met: average difference increased.")
            df_drop[target_column] = previous_imputed_values
            break

        previous_imputed_values = current_imputed_values
        average_difference = current_difference
        iteration += 1

    # 将标签编码器还原
    for column in df_drop.columns:
        df_drop[column] = label_encoders[column].inverse_transform(df_drop[column])

    df_copy[target_column] = df_drop[target_column]
    df_copy.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    unrelated_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, unrelated_column)