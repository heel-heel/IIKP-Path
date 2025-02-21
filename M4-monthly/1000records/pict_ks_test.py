import pandas as pd
import numpy as np
import os
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

df1 = pd.read_csv('clean.csv')
dataset1 = df1['V2']

save_path='./ks_test/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 设置模型名字列表
model_names = ['mean','median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'randomforest', 'xgboost', 'gan', 'midae']
percentages = [10, 30, 50, 70, 90]
csv_files = [f'dirty-{model}-{p}.csv' for model in model_names for p in percentages]


ks_results = []

for file in csv_files:
    model = file.split('-')[1]
    read_path = f'null-{model}'
    df = pd.read_csv(os.path.join(read_path,file))
    dataset = df['V2']
    stat, p_value = ks_2samp(dataset1, dataset)
    ks_results.append((file, stat, p_value))

results_df = pd.DataFrame(ks_results, columns=['File', 'Statistic', 'P-Value'])
results_df['Same Distribution'] = results_df['P-Value'] > 0.05

results_df.to_csv(os.path.join(save_path,'ks_test_results.csv'), index=False)
print("KS Test results have been saved to 'ks_test_results.csv'")

plt.figure(figsize=(12, 8))
sorted_dataset1 = np.sort(dataset1)
yvals1 = np.arange(1, len(sorted_dataset1) + 1) / len(sorted_dataset1)
plt.plot(sorted_dataset1, yvals1, label='clean.csv', linewidth=2, color='black')

for file in csv_files:
    if '50' in file:
        model_name = file.split('-')[1]
        read_path = f'null-{model_name}'
        df = pd.read_csv(os.path.join(read_path,file))
        dataset = df['V2']
        sorted_dataset = np.sort(dataset)
        yvals = np.arange(1, len(sorted_dataset) + 1) / len(sorted_dataset)
        plt.plot(sorted_dataset, yvals, label=model_name, linewidth=1.5)

plt.legend(loc='lower right', fontsize='small')
plt.title('Cumulative Distribution Functions (CDFs)')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.savefig(os.path.join(save_path,'cdf_plot.png'))
plt.show()