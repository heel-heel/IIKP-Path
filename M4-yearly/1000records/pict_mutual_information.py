import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt

save_path='./mutual_information/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 1. 导入CSV文件
original_data_path = 'clean.csv'  # 原始数据文件路径
original_df = pd.read_csv(original_data_path)
original_df = original_df.drop(columns=['V1'])

# 2. 定义填补方法和缺失率
model_names = ['mean', 'median', 'knn', 'hdi', 'mice', 'iim', 'si', 'randomforest', 'xgboost', 'gan', 'midae']
percentages = [10, 30, 50, 70, 90]
csv_files = [f'dirty-{model}-{p}.csv' for model in model_names for p in percentages]

# 3. 计算互信息
def calculate_mutual_information(original_df, imputed_df, column, discrete_features=None):
    """
    计算指定列的互信息。
    :param original_df: 原始数据的DataFrame
    :param imputed_df: 填补数据的DataFrame
    :param column: 指定列名
    :param discrete_features: 是否每个特征是离散的（布尔数组，默认为None，自动判断）
    :return: 指定列的互信息
    """
    # 判断特征是否为离散特征
    if discrete_features is None:
        dtype = original_df[column].dtype
        is_discrete = isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_integer_dtype(dtype)
    else:
        is_discrete = discrete_features[original_df.columns.get_loc(column)]

    # 选择合适的函数计算MI
    if is_discrete:
        mi = mutual_info_classif(original_df[[column]], imputed_df[column], discrete_features=[True], random_state=42)[0]
    else:
        mi = mutual_info_regression(original_df[[column]], imputed_df[column], discrete_features=[False], random_state=42)[0]

    return mi

# 4. 计算所有填补方法和缺失率下的互信息
results = []
for file in csv_files:
    model = file.split('-')[1]
    imputed_df = pd.read_csv(os.path.join(f'./null-{model}/', file)).drop(columns=['V1'])
    assert list(original_df.columns) == list(imputed_df.columns), "原始数据和填补数据的特征列必须一致"
    mi_v2 = calculate_mutual_information(original_df, imputed_df, column='V2')
    results.append((file, mi_v2))

# 5. 输出和可视化结果
# 打印每个文件的互信息
for file, mi in results:
    print(f"File: {file}, Mutual Information for V2: {mi:.4f}")

# 将结果导出到CSV文件
results_df = pd.DataFrame(results, columns=['File', 'Mutual_Information_V2'])
results_df.to_csv(os.path.join(save_path,'mutual_information_results.csv'), index=False)

# 可视化
fig, ax = plt.subplots(figsize=(15, 8))

# 手动定义颜色
model_colors = {
    'mean': 'blue',
    'median': 'orange',
    'knn': 'green',
    'hdi': 'red',
    'mice': 'purple',
    'iim': 'brown',
    'si': 'pink',
    'randomforest': 'gray',
    'xgboost': 'olive',
    'gan': 'cyan',
    'midae': 'magenta',
}

# 绘制柱形图
width = 1.5  # 柱子的宽度
for idx, model in enumerate(model_names):
    model_results = [mi for file, mi in results if model in file]
    ax.bar([p + idx * width for p in percentages], model_results, width=width, label=model, color=model_colors[model])

# 添加图例和标签
ax.set_xlabel("Missing Data Percentage")
ax.set_ylabel("Mutual Information")
ax.set_title("Mutual Information between Original and Imputed Data")
ax.set_xticks([p + (len(model_names) - 1) * width / 2 for p in percentages])
ax.set_xticklabels(percentages)
ax.legend()
ax.grid(False)

# 保存图片
plt.tight_layout()
plt.savefig(os.path.join(save_path,'mutual_information_plot.png'))
plt.show()