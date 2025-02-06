import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

model_names = ['mean', 'median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'randomforest', 'xgboost']
percentages = [10, 30, 50, 70, 90]
csv_files = [f'dirty-{model}-{p}.csv' for model in model_names for p in percentages] + ['clean.csv']

save_path = './regre-analysis/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
results = pd.DataFrame(columns=['file_name', 'Max_Correlation', 'Min_Correlation', 'Avg_Correlation'])

for file_name in csv_files:
    try:
        df = pd.read_csv(file_name)
        df_numeric = df.drop(columns=['V1'], errors='ignore')
        correlation_matrix = df_numeric.corr()
        if 'V2' in correlation_matrix.columns:
            target_correlations = correlation_matrix['V2'].drop('V2', errors='ignore')

            max_corr = target_correlations.max()
            min_corr = target_correlations.min()
            avg_corr = target_correlations.mean()

            new_row = pd.DataFrame({
                'file_name': [file_name],
                'Max_Correlation': [max_corr],
                'Min_Correlation': [min_corr],
                'Avg_Correlation': [avg_corr]
            })

            results = pd.concat([results, new_row], ignore_index=True)
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

results.to_csv(os.path.join(save_path, 'correlation_results.csv'), index=False)
print("相关性结果已保存到 correlation_results.csv")

results['model'] = results['file_name'].apply(lambda x: 'clean' if x == 'clean.csv' else x.split('-')[1])
results['percentage'] = results['file_name'].apply(lambda x: 0 if x == 'clean.csv' else int(x.split('-')[2].replace('.csv', '')))
heatmap_data = results[results['model'] != 'clean'].pivot(index='percentage', columns='model', values='Avg_Correlation')
heatmap_data = heatmap_data.reindex(columns=model_names)
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap='Blues', fmt=".4f", linewidths=.5)
plt.title('Average Correlation Heatmap')
plt.xlabel('Model')
plt.ylabel('Missing Percentage')
plt.xticks(rotation=45, ha='right')
plt.show()