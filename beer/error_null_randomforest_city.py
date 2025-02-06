import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

save_path='./null-city-randomforest/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

file_prefixes = ['dirty-city-10', 'dirty-city-30', 'dirty-city-50', 'dirty-city-70', 'dirty-city-90']

for prefix in file_prefixes:
    df = pd.read_csv(f'{prefix}.csv')
    df_copy=pd.read_csv(f'{prefix}.csv')

    # 假设 'city' 是目标变量
    target_variable = 'city'
    categorical_cols = df.columns.drop([target_variable])


    # 将有缺失值和没有缺失值的行分为两部分
    mask = df[target_variable].isnull()

    # 创建标签编码器
    label_encoders = {}
    for column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    df_drop=df.drop(columns=['id'])

    df_complete = df_drop[~mask]
    df_incomplete = df_drop[mask]

    # 构建训练集和测试集
    X_train = df_complete.drop(target_variable, axis=1)
    y_train = df_complete[target_variable]
    X_test = df_incomplete.drop(target_variable, axis=1)

    # 训练随机森林模型
    rfc = RandomForestRegressor(n_estimators=100, random_state=0)
    rfc.fit(X_train, y_train)

    # 预测缺失值
    y_pred = rfc.predict(X_test)

    # 找到与y_pred最接近的city列的编码值
    # 假设y_pred是连续值，我们需要找到与之最接近的整数编码
    y_pred_rounded = [round(value) for value in y_pred]

    # 填充缺失值
    df_drop.loc[df_incomplete.index, target_variable] = y_pred_rounded

    # 解码分类变量
    for column in df_drop.columns:
        df_drop[column] = label_encoders[column].inverse_transform(df_drop[column])

    df_copy['city']=df_drop['city']
    # 导出填充后的CSV文件
    export_filename = f'dirty-city-randomforest-{prefix[11:]}.csv'
    df_copy.to_csv(os.path.join(save_path,export_filename), index=False)