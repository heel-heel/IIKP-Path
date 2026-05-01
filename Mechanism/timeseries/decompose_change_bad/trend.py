import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

datasets = {
    #"ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},#24 #(1, 5, 50000)(2, 5, 50000)(1, 10, 50000)
    #"ETTm1": {"target_column": "OT", "nonnumerical_column": "date"}, # 96 #(1, 3, 50000)(2, 5, 50000)(1, 10, 50000)
    #"Illness": {"target_column": "OT", "nonnumerical_column": "date"},#52 #(300000, 350000, 50000)(100000, 200000, 50000)(50000, 500000, 50000)
    #"Exchange": {"target_column": "OT", "nonnumerical_column": "date"},#7 #(0.0001, 0.0005, 50000)(0.001, 0.01, 50000)(0.01, 0.1, 50000)
    "Weather": {"target_column": "OT", "nonnumerical_column": "date"}#6 #(0.05, 0.1, 50000)(3, 4, 50000)(5, 50, 50000)

}
target_corrs = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
#target_corrs = [0.70]
cycle = 6
half_cycle = cycle // 2

for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]

    base_path = "../../../Datasets"
    output_path = os.path.join(base_path, dataset, "Mechanism", "timeseries", "decompose_change_bad", "trend")
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

    def add_noise_to_seasonal_and_resid(target_corr_seasonal, target_corr_resid, fraction=0.5):
        np.random.seed(0)
        n = len(df['seasonal'])
        indices = np.random.choice(n, size=int(n * fraction), replace=False)
        best_std_dev_seasonal = None
        best_corr_seasonal = None
        min_diff_seasonal = float('inf')
        std_devs_seasonal = np.linspace(0.07, 0.09, 10000)

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

        best_std_dev_resid = None
        best_corr_resid = None
        min_diff_resid = float('inf')
        std_devs_resid = np.linspace(4.5, 4.7, 10000)

        for std_dev in std_devs_resid:
            noisy_resid = df['resid'].copy()
            noisy_resid.iloc[indices] += np.random.normal(0, std_dev, size=len(indices))
            corr_resid = np.corrcoef(df['resid'], noisy_resid)[0, 1]
            diff = abs(corr_resid - target_corr_resid)
            if diff < min_diff_resid:
                min_diff_resid = diff
                best_std_dev_resid = std_dev
                best_corr_resid = corr_resid
                #print("resid:", std_dev, diff, corr_resid)
            if min_diff_resid < 1e-5:
                break

        return best_std_dev_seasonal, best_corr_seasonal, noisy_seasonal, best_std_dev_resid, best_corr_resid, noisy_resid

    target_corr_seasonal = 0.5
    target_corr_resid = 0.5
    best_std_dev_seasonal, best_corr_seasonal, noisy_seasonal, best_std_dev_resid, best_corr_resid, noisy_resid = add_noise_to_seasonal_and_resid(
        target_corr_seasonal, target_corr_resid)
    print(f"最佳seasonal标准差：{best_std_dev_seasonal}, 最佳seasonal相关系数：{best_corr_seasonal}")
    print(f"最佳resid标准差：{best_std_dev_resid}, 最佳resid相关系数：{best_corr_resid}")

    def generate_dirty_trend_and_corr(std_dev, fraction=0.5):
        np.random.seed(0)
        n = len(df['trend'])
        indices = np.random.choice(n, size=int(n * fraction), replace=False)
        intercept = np.random.normal(0, std_dev)
        dirty_trend = df['trend'].copy()
        dirty_trend.iloc[indices] += intercept + np.random.normal(0, std_dev, size=len(indices))
        dirty_data = dirty_trend + noisy_seasonal + noisy_resid
        dirty_df = pd.DataFrame({
            target_column: dirty_data,
            'trend': dirty_trend,
            'seasonal': noisy_seasonal,
            'resid': noisy_resid
        })

        dirty_df.iloc[:half_cycle, 0] = df[target_column].iloc[:half_cycle]
        dirty_df.iloc[-half_cycle:, 0] = df[target_column].iloc[-half_cycle:]
        dirty_df['trend'] = dirty_df['trend'].fillna(0)
        corr_trend = np.corrcoef(df['trend'], dirty_df['trend'])[0, 1]
        return corr_trend, dirty_df

    std_devs = np.linspace(5, 30, 50000)
    results = pd.DataFrame(columns=['Target Correlation', 'Best Standard Deviation', 'Best Correlation', 'Original vs Generated Correlation'])

    for target_corr in target_corrs:
        best_std_dev = None
        best_corr = None
        min_diff = float('inf')

        for std_dev in std_devs:
            corr_trend, _ = generate_dirty_trend_and_corr(std_dev, fraction=0.5)
            diff = abs(corr_trend - target_corr)
            if diff < min_diff:
                min_diff = diff
                best_std_dev = std_dev
                best_corr = corr_trend
            if min_diff < 1e-5:
                break

        if best_std_dev is None:
            print(f"未找到使相关系数接近{target_corr}的标准差")
        else:
            # 使用最佳标准差生成dirty数据集
            _, dirty_df = generate_dirty_trend_and_corr(best_std_dev, fraction=0.5)
            corr_clean_dirty = np.corrcoef(data, dirty_df[target_column])[0, 1]

            print("--------------------------------")
            print(f"目标相关系数：{target_corr}")
            print(f"最佳标准差：{best_std_dev}")
            print(f"最佳相关系数：{best_corr}")
            print(f"修改后的数据与clean的相关系数：{corr_clean_dirty}")

            new_row = pd.DataFrame({
                'Target Correlation': [target_corr],
                'Best Standard Deviation': [best_std_dev],
                'Best Correlation': [best_corr],
                'Original vs Generated Correlation': [corr_clean_dirty]
            })

            results = pd.concat([results, new_row], ignore_index=True)
            df_copy[target_column] = dirty_df[target_column]
            df_copy.to_csv(os.path.join(output_path, f'dirty-trend-50-{int(target_corr * 100)}.csv'), index=False)


            # fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            #
            # df_head = df.iloc[half_cycle:200]
            # dirty_df_head = dirty_df.iloc[half_cycle:200]
            #
            # axs[0].plot(df_head.index, df_head[target_column], label='Clean', color='blue')
            # axs[0].plot(dirty_df_head.index, dirty_df_head[target_column], label='Dirty', color='red')
            # axs[0].set_title('Original Data')
            # axs[0].legend()
            #
            # # Trend
            # axs[1].plot(df_head.index, df_head['trend'], label='Clean', color='blue')
            # axs[1].plot(dirty_df_head.index, dirty_df_head['trend'], label='Dirty', color='red')
            # axs[1].set_title('Trend')
            # axs[1].legend()
            #
            # # Seasonal
            # axs[2].plot(df_head.index, df_head['seasonal'], label='Clean', color='blue')
            # axs[2].plot(dirty_df_head.index, dirty_df_head['seasonal'], label='Dirty', color='red')
            # axs[2].set_title('Seasonal')
            # axs[2].legend()
            #
            # # Resid
            # axs[3].plot(df_head.index, df_head['resid'], label='Clean', color='blue')
            # axs[3].plot(dirty_df_head.index, dirty_df_head['resid'], label='Dirty', color='red')
            # axs[3].set_title('Residual')
            # axs[3].legend()
            #
            # plt.tight_layout()
            # plt.savefig(os.path.join(output_fig_path, f'decompose_change_bad_trend_{int(target_corr * 100)}.png'))
            # plt.close()

    results.to_csv(os.path.join(output_path, 'decompose_change_bad_trend_results.csv'), index=False)
    print(f"Results for {dataset} saved to '{output_path}/decompose_change_bad_trend_results.csv'")