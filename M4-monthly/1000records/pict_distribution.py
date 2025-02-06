import pandas as pd
import numpy as np
import os
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# 读取clean.csv文件
df1 = pd.read_csv('clean.csv')
dataset1 = df1['V2']

save_path='./ks_test/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 设置模型名字列表
model_names = ['mean', 'knn', 'mice', 'iim', 'si', 'randomforest', 'xgboost']
percentages = [10, 30, 50, 70, 90]

# 生成其他CSV文件的名称
csv_files = [f'dirty-{model}-{p}.csv' for model in model_names for p in percentages]

# 存储KS检验结果
ks_results = []

# 逐个读取并进行KS检验
for file in csv_files:
    df = pd.read_csv(file)
    dataset = df['V2']

    # 进行KS检验
    stat, p_value = ks_2samp(dataset1, dataset)

    # 存储结果
    ks_results.append((file, stat, p_value))

# 创建结果DataFrame
results_df = pd.DataFrame(ks_results, columns=['File', 'Statistic', 'P-Value'])

# 添加是否符合同一分布的列
results_df['Same Distribution'] = results_df['P-Value'] > 0.05

# 输出结果到CSV文件
results_df.to_csv(os.path.join(save_path,'ks_test_results.csv'), index=False)

# 打印结果
print("KS Test results have been saved to 'ks_test_results.csv'")

# 绘制累积分布函数（CDF）图像
plt.figure(figsize=(12, 8))

# 绘制clean.csv的CDF
sorted_dataset1 = np.sort(dataset1)
yvals1 = np.arange(1, len(sorted_dataset1) + 1) / len(sorted_dataset1)
plt.plot(sorted_dataset1, yvals1, label='clean.csv', linewidth=2, color='black')

# 绘制50%的其他数据集的CDF
for file in csv_files:
    if '50' in file:  # 选择50%的数据集
        model_name = file.split('-')[1]  # 提取模型名字
        df = pd.read_csv(file)
        dataset = df['V2']
        sorted_dataset = np.sort(dataset)
        yvals = np.arange(1, len(sorted_dataset) + 1) / len(sorted_dataset)
        plt.plot(sorted_dataset, yvals, label=model_name, linewidth=1.5)

# 添加图例
plt.legend(loc='lower right', fontsize='small')

# 添加标题和轴标签
plt.title('Cumulative Distribution Functions (CDFs)')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')

# 显示图像
plt.show()