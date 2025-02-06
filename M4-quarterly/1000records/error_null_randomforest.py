import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor

save_path='./null-randomforest/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 文件名前缀列表，包含特定的前缀
file_prefixes = ['dirty-10', 'dirty-30', 'dirty-50', 'dirty-70', 'dirty-90']

# 遍历所有文件名前缀
for prefix in file_prefixes:
    # 读取CSV文件
    df = pd.read_csv(f'{prefix}.csv')

    # 保留V1列的数据
    v1_data = df['V1'].copy()

    # 去掉V1列
    df.drop('V1', axis=1, inplace=True)

    # 将有缺失值和没有缺失值的行分为两部分
    mask = df['V2'].isnull()  # 找出V2列中有缺失值的行
    df_complete = df[~mask]  # 没有缺失值的部分
    df_incomplete = df[mask]  # 有缺失值的部分

    # 构建训练集和测试集
    X_train = df_complete.drop('V2', axis=1)  # 非缺失值作为训练特征
    y_train = df_complete['V2']  # 非缺失值作为训练标签
    X_test = df_incomplete.drop('V2', axis=1)  # 缺失值作为测试特征

    # 训练随机森林模型
    rfc = RandomForestRegressor(n_estimators=100, random_state=0)
    rfc.fit(X_train, y_train)

    # 预测缺失值
    y_pred = rfc.predict(X_test)

    # 填充缺失值
    df.loc[mask, 'V2'] = y_pred

    # 将V1列添加回df中，并置于最前面
    df = pd.concat([v1_data.to_frame(), df], axis=1)
    df.columns.values[0] = 'V1'  # 重命名第一列为'V1'

    # 导出填充后的CSV文件
    export_filename = f'dirty-randomforest-{prefix[6:]}.csv'
    df.to_csv(os.path.join(save_path,export_filename), index=False)