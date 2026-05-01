import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

target_corrs = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
#target_corrs = [0.98]
cycle = 12
half_cycle = cycle // 2


output_path = os.path.join("decompose_change_good_results", "seasonal")
if not os.path.exists(output_path):
    os.makedirs(output_path)
output_fig_path = os.path.join(output_path, 'fig')
if not os.path.exists(output_fig_path):
    os.makedirs(output_fig_path)

target_column = 'y_combined'
input_clean_file = os.path.join("../generate_timeseries_results", "synthetic_time_series_balanced.csv")
clean_df = pd.read_csv(input_clean_file)
df_copy = clean_df.copy()
clean_data = clean_df[target_column]

seasonal_decomp = seasonal_decompose(clean_data, model="additive", period=cycle)
clean_df['trend'] = seasonal_decomp.trend
clean_df['seasonal'] = seasonal_decomp.seasonal
clean_df['resid'] = seasonal_decomp.resid

clean_df['trend'] = clean_df['trend'].fillna(0)
clean_df['seasonal'] = clean_df['seasonal'].fillna(0)
clean_df['resid'] = clean_df['resid'].fillna(0)


def generate_dirty_seasonal_and_corr(std_dev, fraction=0.5):
    np.random.seed(0)
    n = len(clean_df['seasonal'])
    indices = np.random.choice(n, size=int(n * fraction), replace=False)
    intercept = np.random.normal(0, std_dev)
    dirty_seasonal = clean_df['seasonal'].copy()
    dirty_seasonal.iloc[indices] += intercept + np.random.normal(0, std_dev, size=len(indices))
    dirty_data = dirty_seasonal + clean_df['trend'] + clean_df['resid']
    dirty_df = pd.DataFrame({target_column: dirty_data, 'trend': clean_df['trend'], 'seasonal': dirty_seasonal, 'resid': clean_df['resid']})

    dirty_df.iloc[:half_cycle, 0] = clean_df[target_column].iloc[:half_cycle]
    dirty_df.iloc[-half_cycle:, 0] = clean_df[target_column].iloc[-half_cycle:]
    dirty_df['seasonal'] = dirty_df['seasonal'].fillna(0)
    corr_trend = np.corrcoef(clean_df['seasonal'], dirty_df['seasonal'])[0, 1]
    return corr_trend, dirty_df


std_devs = np.linspace(0.2, 1.5, 50000)
results = pd.DataFrame(columns=['Target Correlation', 'Best Standard Deviation', 'Best Correlation', 'Original vs Generated Correlation'])

for target_corr in target_corrs:
    best_std_dev = None
    best_corr = None
    min_diff = float('inf')

    for std_dev in std_devs:
        corr_trend, _ = generate_dirty_seasonal_and_corr(std_dev, fraction=0.5)
        diff = abs(corr_trend - target_corr)
        if diff < min_diff:
            min_diff = diff
            best_std_dev = std_dev
            best_corr = corr_trend
        if min_diff < 1e-5:
            break

    if best_std_dev is None:
        print(f"Standard deviation that makes the correlation coefficient close to {target_corr} was not found")
    else:
        _, dirty_df = generate_dirty_seasonal_and_corr(best_std_dev, fraction=0.5)
        corr_trend2 = np.corrcoef(clean_data, dirty_df[target_column])[0, 1]

        print("--------------------------------")
        print(f"target_corr: {target_corr}")
        print(f"best_std_dev: {best_std_dev}")
        print(f"best_corr: {best_corr}")
        print(f"corr_trend2: {corr_trend2}")

        new_row = pd.DataFrame({
            'Target Correlation': [target_corr],
            'Best Standard Deviation': [best_std_dev],
            'Best Correlation': [best_corr],
            'Original vs Generated Correlation': [corr_trend2]
        })

        results = pd.concat([results, new_row], ignore_index=True)

        df_copy[target_column] = dirty_df[target_column]
        df_copy.to_csv(os.path.join(output_path, f'dirty-seasonal-50-{int(target_corr * 100)}.csv'), index=False)


results.to_csv(os.path.join(output_path, 'decompose_change_good_seasonal_results.csv'), index=False)
print(f"Results has saved to '{output_path}/decompose_change_good_seasonal_results.csv'")