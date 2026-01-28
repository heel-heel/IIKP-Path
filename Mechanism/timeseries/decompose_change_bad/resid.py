import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

datasets = {
    #"M4-Daily": {"target_column": "V2", "nonnumerical_column": "V1"},#best trend std_dev: 8840.883441192018, best trend correlation: 0.4999999852810793;best seasonality std_dev: 676.265375219646, best seasonality correlation: 0.4999946844668556
    #"M4-Weekly": {"target_column": "V2", "nonnumerical_column": "V1"},#best trend std_dev: 5547.630952619053, best trend correlation: 0.499999691269482;best seasonality std_dev: 2386.907738154763, best seasonality correlation: 0.4999909863025729
    #"M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},#best trend std_dev: 5055.521110422209, best trend correlation: 0.499992240873415;best seasonality std_dev: 935.0217041259268, best seasonality correlation: 0.5000051966048961
    #"M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},#best trend std_dev: 5428.728574571492, best trend correlation: 0.5000040365258018;best seasonality std_dev: 829.5814889418142, best seasonality correlation: 0.5000006929509552
    #"M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"},#best trend std_dev: 4845.016900338007, best trend correlation: 0.49999450521906386;best seasonality std_dev: 734.9512958189711, best seasonality correlation: 0.49999098268787284

    #"ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},#24 #(10, 20, 10000)(1, 5, 50000)(0.1, 5, 100000)
    #"ETTm1": {"target_column": "OT", "nonnumerical_column": "date"},#96 #(15, 20, 50000)(1, 3, 50000)(0.1, 2, 50000)
    #"Illness": {"target_column": "OT", "nonnumerical_column": "date"},#52 #(700000, 800000, 50000)(20000, 300000, 50000)(10000, 100000, 50000)
    #"Exchange": {"target_column": "OT", "nonnumerical_column": "date"},#7 #
    "Weather": {"target_column": "OT", "nonnumerical_column": "date"}#6 #(80, 100, 50000)(0.08, 1, 50000)(0.1, 2, 50000)
}
target_corrs = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
#target_corrs = [0.98]
cycle = 6
half_cycle = cycle // 2

for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]
    base_path = "../../../Datasets"
    output_path = os.path.join(base_path, dataset, "Mechanism", "timeseries", "decompose_change_bad", "resid")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_fig_path = os.path.join(output_path, 'fig')
    if not os.path.exists(output_fig_path):
        os.makedirs(output_fig_path)

    input_clean_file = os.path.join(base_path, dataset, "clean.csv")
    df = pd.read_csv(input_clean_file)
    df_copy = df.copy()
    data = df[target_column]

    seasonal_decomp = seasonal_decompose(data, model="additive", period=cycle)
    df['trend'] = seasonal_decomp.trend
    df['seasonal'] = seasonal_decomp.seasonal
    df['resid'] = seasonal_decomp.resid

    df['trend'] = df['trend'].fillna(0)
    df['seasonal'] = df['seasonal'].fillna(0)
    df['resid'] = df['resid'].fillna(0)

    def add_noise_to_trend_and_seasonal(target_corr_trend, target_corr_seasonal, fraction=0.5):
        np.random.seed(0)
        n = len(df['trend'])
        indices = np.random.choice(n, size=int(n * fraction), replace=False)

        best_std_dev_trend = None
        best_corr_trend = None
        min_diff_trend = float('inf')
        std_devs_trend = np.linspace(80, 100, 50000)

        for std_dev in std_devs_trend:
            noisy_trend = df['trend'].copy()
            noisy_trend.iloc[indices] += np.random.normal(0, std_dev, size=len(indices))
            corr_trend = np.corrcoef(df['trend'], noisy_trend)[0, 1]
            diff = abs(corr_trend - target_corr_trend)
            if diff < min_diff_trend:
                min_diff_trend = diff
                best_std_dev_trend = std_dev
                best_corr_trend = corr_trend
                #print("trend:", std_dev, diff, corr_trend)
            if min_diff_trend < 1e-5:
                break

        best_std_dev_seasonal = None
        best_corr_seasonal = None
        min_diff_seasonal = float('inf')
        std_devs_seasonal = np.linspace(0.08, 1, 50000)

        for std_dev in std_devs_seasonal:
            noisy_seasonal = df['seasonal'].copy()
            noisy_seasonal.iloc[indices] += np.random.normal(0, std_dev, size=len(indices))
            corr_seasonal = np.corrcoef(df['seasonal'], noisy_seasonal)[0, 1]
            diff = abs(corr_seasonal - target_corr_seasonal)
            if diff < min_diff_seasonal:
                min_diff_seasonal = diff
                best_std_dev_seasonal = std_dev
                best_corr_seasonal = corr_seasonal
                #print("seasonal:", std_dev, diff, corr_seasonal)
            if min_diff_seasonal < 1e-5:
                break
        return best_std_dev_trend, best_corr_trend, noisy_trend, best_std_dev_seasonal, best_corr_seasonal, noisy_seasonal

    target_corr_trend = 0.5
    target_corr_seasonal = 0.5
    best_std_dev_trend, best_corr_trend, noisy_trend, best_std_dev_seasonal, best_corr_seasonal, noisy_seasonal = add_noise_to_trend_and_seasonal(
        target_corr_trend, target_corr_seasonal)
    print(f"best trend std_dev：{best_std_dev_trend}, best trend correlation：{best_corr_trend}")
    print(f"best seasonality std_dev：{best_std_dev_seasonal}, best seasonality correlation：{best_corr_seasonal}")

    def generate_dirty_resid_and_corr(std_dev, fraction=0.5):
        np.random.seed(0)
        n = len(df['resid'])
        indices = np.random.choice(n, size=int(n * fraction), replace=False)
        intercept = np.random.normal(0, std_dev)
        dirty_resid = df['resid'].copy()
        dirty_resid.iloc[indices] += intercept + np.random.normal(0, std_dev, size=len(indices))
        dirty_data = dirty_resid + noisy_trend + noisy_seasonal
        dirty_df = pd.DataFrame({
            target_column: dirty_data,
            'trend': noisy_trend,
            'seasonal': noisy_seasonal,
            'resid': dirty_resid
        })

        dirty_df.iloc[:half_cycle, 0] = df[target_column].iloc[:half_cycle]
        dirty_df.iloc[-half_cycle:, 0] = df[target_column].iloc[-half_cycle:]
        dirty_df['resid'] = dirty_df['resid'].fillna(0)
        corr_resid = np.corrcoef(df['resid'], dirty_df['resid'])[0, 1]
        return corr_resid, dirty_df

    std_devs = np.linspace(0.1, 2, 50000)
    results = pd.DataFrame(columns=['Target Correlation', 'Best Standard Deviation', 'Best Correlation',
                                    'Original vs Generated Correlation'])
    for target_corr in target_corrs:
        best_std_dev = None
        best_corr = None
        min_diff = float('inf')

        for std_dev in std_devs:
            corr_resid, _ = generate_dirty_resid_and_corr(std_dev, fraction=0.5)
            diff = abs(corr_resid - target_corr)
            if diff < min_diff:
                min_diff = diff
                best_std_dev = std_dev
                best_corr = corr_resid
            if min_diff < 1e-5:
                break

        if best_std_dev is None:
            print(f"Standard deviation that makes the correlation coefficient close to {target_corr} was not found")
        else:
            _, dirty_df = generate_dirty_resid_and_corr(best_std_dev, fraction=0.5)
            corr_clean_dirty = np.corrcoef(data, dirty_df[target_column])[0, 1]

            print("--------------------------------")
            print(f"target_corr: {target_corr}")
            print(f"vest_std_dev: {best_std_dev}")
            print(f"best_corr: {best_corr}")
            print(f"corr_clean_dirty: {corr_clean_dirty}")

            new_row = pd.DataFrame({
                'Target Correlation': [target_corr],
                'Best Standard Deviation': [best_std_dev],
                'Best Correlation': [best_corr],
                'Original vs Generated Correlation': [corr_clean_dirty]
            })
            results = pd.concat([results, new_row], ignore_index=True)
            df_copy[target_column] = dirty_df[target_column]
            df_copy.to_csv(os.path.join(output_path, f'dirty-resid-50-{int(target_corr * 100)}.csv'), index=False)


            fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

            df_head = df.iloc[half_cycle:200]
            dirty_df_head = dirty_df.iloc[half_cycle:200]

            axs[0].plot(df_head.index, df_head[target_column], label='Clean', color='blue')
            axs[0].plot(dirty_df_head.index, dirty_df_head[target_column], label='Dirty', color='red')
            axs[0].set_title('Original Data')
            axs[0].legend()

            # Trend
            axs[1].plot(df_head.index, df_head['trend'], label='Clean', color='blue')
            axs[1].plot(dirty_df_head.index, dirty_df_head['trend'], label='Dirty', color='red')
            axs[1].set_title('Trend')
            axs[1].legend()

            # Seasonality
            axs[2].plot(df_head.index, df_head['seasonal'], label='Clean', color='blue')
            axs[2].plot(dirty_df_head.index, dirty_df_head['seasonal'], label='Dirty', color='red')
            axs[2].set_title('Seasonal')
            axs[2].legend()

            # Residual
            axs[3].plot(df_head.index, df_head['resid'], label='Clean', color='blue')
            axs[3].plot(dirty_df_head.index, dirty_df_head['resid'], label='Dirty', color='red')
            axs[3].set_title('Residual')
            axs[3].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_fig_path, f'decompose_change_bad_resid_{int(target_corr * 100)}.png'))
            plt.close()

    results.to_csv(os.path.join(output_path, 'decompose_change_bad_resid_results.csv'), index=False)
    print(f"Results for {dataset} saved to '{output_path}/decompose_change_bad_resid_results.csv'")