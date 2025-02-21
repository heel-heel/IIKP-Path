import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import os

save_path='./2w_distance/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def calculate_2wasserstein_distance(clean_series, dirty_series):
    distance = wasserstein_distance(clean_series, dirty_series)
    return distance

model_names = ['mean', 'median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'randomforest', 'xgboost', 'gan', 'midae']
percentages = [10, 30, 50, 70, 90]
csv_files = [f'dirty-{model}-{p}.csv' for model in model_names for p in percentages]

clean_data = pd.read_csv("clean.csv")
clean_v2 = clean_data['V2'].values

mean = np.mean(clean_v2)
std = np.std(clean_v2)
clean_v2_standardized = (clean_v2 - mean) / std

results = []
for file in csv_files:
    model = file.split('-')[1]
    percentage = file.split('-')[2].split('.')[0]
    dirty_path = os.path.join(f'./null-{model}', file)
    dirty_data = pd.read_csv(dirty_path)
    dirty_v2 = dirty_data['V2'].values
    dirty_v2_standardized = (dirty_v2 - mean) / std

    distance = calculate_2wasserstein_distance(clean_v2_standardized, dirty_v2_standardized)

    results.append({
        'file':file,
        '2-Wasserstein Distance': distance
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_path,"wasserstein_distances.csv"), index=False)
print("2-Wasserstein Distance results saved to 'wasserstein_distances.csv'")