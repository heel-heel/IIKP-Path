import pandas as pd
import numpy as np
import os

model_names = ['mean', 'median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'randomforest', 'xgboost']
percentages = [10, 30, 50, 70, 90]
csv_files = [f'dirty-{model}-{p}.csv' for model in model_names for p in percentages]+['clean.csv']

save_path = './regre-analysis/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

clean_df = pd.read_csv('clean.csv')
results = pd.DataFrame(columns=['file_name', 'RMSE', 'MAE'])

for file_name in csv_files:
    if file_name != 'clean.csv':
        try:
            dirty_df = pd.read_csv(file_name)
            rmse = np.sqrt(np.mean((dirty_df['V2'] - clean_df['V2']) ** 2))
            mae = np.mean(np.abs(dirty_df['V2'] - clean_df['V2']))
            new_row = pd.DataFrame({
                'file_name': [file_name],
                'RMSE': [rmse],
                'MAE': [mae]
            })
            results = pd.concat([results, new_row], ignore_index=True)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

results.to_csv(os.path.join(save_path, 'distance_results.csv'), index=False)
print("评估结果已保存到 distance_results.csv")