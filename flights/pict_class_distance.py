import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# 定义模型名称和缺失比例
model_names = ['mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'randomforest', 'xgboost', 'gan', 'midae']
percentages = [10, 30, 50, 70, 90]
csv_files = [f'dirty-flight-{model}-{p}.csv' for model in model_names for p in percentages]
csv_files.append("clean.csv")

save_path='./class_quality/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


results = []
for file in csv_files:
    print(f"正在计算：{file}")
    if file == "clean.csv":
        data = pd.read_csv(file)
    else:
        model = file.split('-')[2]
        read_path = f'null-flight-{model}'
        data = pd.read_csv(os.path.join(read_path,file))

    #data['ibu'] = data['ibu'].fillna(data['ibu'].mean())
    #data['ounces'] = data['ounces'].fillna(data['ounces'].mean())
    #data['abv'] = data['abv'].fillna(data['abv'].mean())

    data.columns = data.columns.str.lower()
    features = data.drop(columns=['flight'])
    target = data['flight']
    features_encoded = pd.get_dummies(features)

    #scaler = StandardScaler()
    #numerical_features = features.select_dtypes(include=[np.number]).columns
    #features_encoded[numerical_features] = scaler.fit_transform(features_encoded[numerical_features])

    # 获取唯一类别
    unique_cities = target.unique()
    # 初始化类内散度矩阵
    within_class_scatter = np.zeros((features_encoded.shape[1], features_encoded.shape[1]))
    # 计算类内散度矩阵
    for city in unique_cities:
        class_data = features_encoded[target == city]
        class_mean = np.mean(class_data, axis=0)
        class_scatter = np.dot((class_data - class_mean).T, (class_data - class_mean))
        within_class_scatter += class_scatter

    overall_mean = np.mean(features_encoded, axis=0)
    # 初始化类间散度矩阵
    between_class_scatter = np.zeros((features_encoded.shape[1], features_encoded.shape[1]))
    # 计算类间散度矩阵
    for city in unique_cities:
        class_data = features_encoded[target == city]
        class_mean = np.mean(class_data, axis=0)
        class_size = len(class_data)
        between_class_scatter += class_size * np.outer((class_mean - overall_mean), (class_mean - overall_mean))

    # 计算类内距离/类间距离/J值
    within_class_distance = np.trace(within_class_scatter)
    between_class_distance = np.trace(between_class_scatter)
    J = between_class_distance / within_class_distance

    print(f"类内距离：{within_class_distance}")
    print(f"类间距离：{between_class_distance}")
    print(f"J值：{J}")

    results.append({
        'File Name': file,
        'Within-Class Distance': within_class_distance,
        'Between-Class Distance': between_class_distance,
        'J Value': J
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_path,"class_distance_results.csv"), index=False)
print("结果已导出到class_distance_results.csv")