import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

file_prefixes = ['dirty-10', 'dirty-30', 'dirty-50', 'dirty-70', 'dirty-90']

for prefix in file_prefixes:
    df = pd.read_csv(f'{prefix}.csv')

    # 假设 'ounces' 是目标变量
    target_variable = 'ounces'
    categorical_cols = df.columns.drop([target_variable])

    # 创建标签编码器
    label_encoders = {}
    for column in categorical_cols:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # 将有缺失值和没有缺失值的行分为两部分
    mask = df[target_variable].isnull()
    df_complete = df[~mask]
    df_incomplete = df[mask]

    # 构建训练集和测试集
    X_train = df_complete.drop(target_variable, axis=1)
    y_train = df_complete[target_variable]
    X_test = df_incomplete.drop(target_variable, axis=1)

    # 训练随机森林模型
    rfc = RandomForestRegressor(n_estimators=100, random_state=0)
    rfc.fit(X_train, y_train)

    # 预测缺失值
    y_pred = rfc.predict(X_test)

    # 填充缺失值
    df.loc[df_incomplete.index, target_variable] = y_pred

    # 解码分类变量
    for column in categorical_cols:
        df[column] = label_encoders[column].inverse_transform(df[column])

    # 导出填充后的CSV文件
    export_filename = f'dirty-randomforest-{prefix[6:]}.csv'
    df.to_csv(export_filename, index=False)