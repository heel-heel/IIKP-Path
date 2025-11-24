import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

datasets = {
    "M4-Daily": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Weekly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"}
}
target_corrs = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
cycle = 12

for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]
    base_path = "../../../Datasets"
    output_path = os.path.join(base_path, dataset, "Mechanism", "timeseries", "decompose_change_good", "trend")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_fig_path = os.path.join(output_path, 'fig')
    if not os.path.exists(output_fig_path):
        os.makedirs(output_fig_path)

    input_clean_file = os.path.join(base_path, dataset, "clean.csv")
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
        n = len(clean_df['trend'])
        indices = np.random.choice(n, size=int(n * fraction), replace=False)
        intercept = np.random.normal(0, std_dev)
        dirty_trend = clean_df['trend'].copy()
        dirty_trend.iloc[indices] += intercept + np.random.normal(0, std_dev, size=len(indices))
        dirty_data = dirty_trend + clean_df['seasonal'] + clean_df['resid']
        dirty_df = pd.DataFrame({target_column: dirty_data, 'trend': dirty_trend, 'seasonal': clean_df['seasonal'], 'resid': clean_df['resid']})

        dirty_df.iloc[:6, 0] = clean_df.iloc[:6, 1]
        dirty_df.iloc[-6:, 0] = clean_df.iloc[-6:, 1]
        dirty_df['trend'] = dirty_df['trend'].fillna(0)
        corr_trend = np.corrcoef(clean_df['trend'], dirty_df['trend'])[0, 1]
        return corr_trend, dirty_df


    std_devs = np.linspace(100, 6000, 59000)
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
            df_copy.to_csv(os.path.join(output_path, f'dirty-trend-50-{int(target_corr * 100)}.csv'), index=False)


            fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

            clean_df_head = clean_df.head(200)
            dirty_df_head = dirty_df.head(200)

            axs[0].plot(clean_df_head.index, clean_df_head[target_column], label='Clean', color='blue')
            axs[0].plot(dirty_df_head.index, dirty_df_head[target_column], label='Dirty', color='red')
            axs[0].set_title('Original Data')
            axs[0].legend()

            # Trend
            axs[1].plot(clean_df_head.index, clean_df_head['trend'], label='Clean', color='blue')
            axs[1].plot(dirty_df_head.index, dirty_df_head['trend'], label='Dirty', color='red')
            axs[1].set_title('Trend')
            axs[1].legend()

            # Seasonality
            axs[2].plot(clean_df_head.index, clean_df_head['seasonal'], label='Clean', color='blue')
            axs[2].plot(dirty_df_head.index, dirty_df_head['seasonal'], label='Dirty', color='red')
            axs[2].set_title('Seasonal')
            axs[2].legend()

            # Residual
            axs[3].plot(clean_df_head.index, clean_df_head['resid'], label='Clean', color='blue')
            axs[3].plot(dirty_df_head.index, dirty_df_head['resid'], label='Dirty', color='red')
            axs[3].set_title('Residual')
            axs[3].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_fig_path, f'decompose_change_good_trend_{int(target_corr * 100)}.png'))
            plt.show()

    results.to_csv(os.path.join(output_path, 'decompose_change_good_trend_results.csv'), index=False)
    print(f"Results for {dataset} saved to '{output_path}/decompose_change_good_trend_results.csv'")