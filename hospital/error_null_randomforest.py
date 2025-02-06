import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

file_prefixes = ['dirty-10', 'dirty-30', 'dirty-50', 'dirty-70', 'dirty-90']

for prefix in file_prefixes:
    df = pd.read_csv(f'{prefix}.csv')

    #df['Score_copy'] = df['Score'].copy()
    Score_copy=df['Score'].copy()
    #df['Score_ori']=df['Score'].copy()
    Score_ori=df['Score'].copy()
    # 去除百分号并转换为数值类型，非数值的转换为NaN
    def clean_score(x):
        if isinstance(x, str) and x not in ['empty', '']:
            return x.strip().replace('%', '')
        return x


    #df['Score_copy'] = df['Score_copy'].apply(clean_score)
    #df['Score_copy'] = pd.to_numeric(df['Score_copy'], errors='coerce')
    Score_copy = Score_copy.apply(clean_score)
    Score_copy = pd.to_numeric(Score_copy, errors='coerce')
    df['Score']=Score_copy


    # 假设 'rate' 是目标变量
    target_variable = 'Score'
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

    # 将Score_copy的值转换为带有百分号的字符串，并替换Score列
    df['Score'] = df['Score'].apply(lambda x: f"{x:.2f}%" if not pd.isnull(x) else x)
    df['Score_ori']=Score_ori
    df['Score'] = df.apply(lambda row: 'empty' if row['Score_ori'] == 'empty' else row['Score'], axis=1)

    # 删除Score_copy列
    #df.drop('Score_copy', axis=1, inplace=True)
    df.drop('Score_ori',axis=1,inplace=True)

    # 导出填充后的CSV文件
    export_filename = f'dirty-randomforest-{prefix[6:]}.csv'
    df.to_csv(export_filename, index=False)