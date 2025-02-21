import pandas as pd
import numpy as np
from scipy.stats import entropy
import os

save_path='./kl_divergence/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def calculate_kl_divergence(clean_series, dirty_series, bins=20):
    combined_data = np.concatenate((clean_series, dirty_series))
    hist, bin_edges = np.histogram(combined_data, bins=bins, density=True)
    clean_hist, _ = np.histogram(clean_series, bins=bin_edges, density=True)
    dirty_hist, _ = np.histogram(dirty_series, bins=bin_edges, density=True)
    clean_prob = clean_hist / np.sum(clean_hist)
    dirty_prob = dirty_hist / np.sum(dirty_hist)
    kl_div = entropy(dirty_prob, clean_prob)
    return kl_div

model_names = ['mean', 'median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'randomforest', 'xgboost', 'gan', 'midae']
percentages = [10, 30, 50, 70, 90]
csv_files = [f'dirty-{model}-{p}.csv' for model in model_names for p in percentages]

clean_data = pd.read_csv("clean.csv")
clean_v2 = clean_data['V2'].values

results = []
for file in csv_files:
    model = file.split('-')[1]
    percentage = file.split('-')[2].split('.')[0]
    dirty_path = os.path.join(f'./null-{model}', file)
    dirty_data = pd.read_csv(dirty_path)
    dirty_v2 = dirty_data['V2'].values

    kl_divergence = calculate_kl_divergence(clean_v2, dirty_v2)

    results.append({
        'file':file,
        'KL_Divergence': kl_divergence
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_path,"kl_divergence_results.csv"), index=False)
print("KL Divergence results saved to 'kl_divergence_results.csv'")