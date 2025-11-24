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
    "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"},

    "M3-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M3-Yearly-history": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M3-Yearly-test": {"target_column": "V2", "nonnumerical_column": "V1"},
}
Imputation_Algorithms = ['mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

cycle = 12
for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]
    base_path = '../../../Datasets'
    input_clean_file = os.path.join(base_path, dataset, "clean.csv")
    output_path = os.path.join(base_path, dataset, "Mechanism", "timeseries", "decompose_basic")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_fig_path = os.path.join(output_path, 'fig')
    if not os.path.exists(output_fig_path):
        os.makedirs(output_fig_path)

    clean_df = pd.read_csv(input_clean_file)
    clean_data = clean_df[target_column]

    dfs_dirty = {}
    for model in Imputation_Algorithms:
        for rate in Missing_rate:
            input_dirty_path = os.path.join(base_path, dataset, "Imputation", f"null-{model}")
            input_dirty_file = f"dirty-{model}-{rate}.csv"
            dfs_dirty[input_dirty_file] = pd.read_csv(os.path.join(input_dirty_path, input_dirty_file))
    data_dirty = {file: df[target_column] for file, df in dfs_dirty.items()}

    decompositions = {'clean': seasonal_decompose(clean_data, model="additive", period=cycle)}
    decompositions.update({file: seasonal_decompose(data, model="additive", period=cycle) for file, data in data_dirty.items()})
    trends = {name: decomp.trend.dropna() for name, decomp in decompositions.items()}
    seasonals = {name: decomp.seasonal.dropna() for name, decomp in decompositions.items()}
    resids = {name: decomp.resid.dropna() for name, decomp in decompositions.items()}

    results = []
    for file, decomp in decompositions.items():
        trend = decomp.trend.dropna()
        seasonal = decomp.seasonal.dropna()
        resid = decomp.resid.dropna()
        mse_trend = np.mean((trend - trends['clean']) ** 2)
        mse_seasonal = np.mean((seasonal - seasonals['clean']) ** 2)
        mse_resid = np.mean((resid - resids['clean']) ** 2)
        rmse_trend = np.sqrt(mse_trend)
        rmse_seasonal = np.sqrt(mse_seasonal)
        rmse_resid = np.sqrt(mse_resid)
        mae_trend = np.mean(np.abs(trend - trends['clean']))
        mae_seasonal = np.mean(np.abs(seasonal - seasonals['clean']))
        mae_resid = np.mean(np.abs(resid - resids['clean']))
        corr_trend = np.corrcoef(trend, trends['clean'])[0, 1]
        corr_seasonal = np.corrcoef(seasonal, seasonals['clean'])[0, 1]
        corr_resid = np.corrcoef(resid, resids['clean'])[0, 1]
        max_abs_error_trend = np.max(np.abs(trend - trends['clean']))
        max_abs_error_seasonal = np.max(np.abs(seasonal - seasonals['clean']))
        max_abs_error_resid = np.max(np.abs(resid - resids['clean']))
        results.append({
            'file': file,
            'MSE_Trend': mse_trend,
            'RMSE_Trend': rmse_trend,
            'MAE_Trend': mae_trend,
            'Corr_Trend': corr_trend,
            'Max_Abs_Error_Trend': max_abs_error_trend,
            'MSE_Seasonal': mse_seasonal,
            'RMSE_Seasonal': rmse_seasonal,
            'MAE_Seasonal': mae_seasonal,
            'Corr_Seasonal': corr_seasonal,
            'Max_Abs_Error_Seasonal': max_abs_error_seasonal,
            'MSE_Residuals': mse_resid,
            'RMSE_Residuals': rmse_resid,
            'MAE_Residuals': mae_resid,
            'Corr_Residuals': corr_resid,
            'Max_Abs_Error_Residuals': max_abs_error_resid
        })

    results_df = pd.DataFrame(results)
    results_file_path = os.path.join(output_path, f"decompose_basic_results.csv")
    results_df.to_csv(results_file_path, index=False)
    print(f"decompose results for {dataset} have been saved to '{results_file_path}'")


    plt.figure(figsize=(14, 12))

    plt.subplot(4, 1, 1)
    plt.plot(clean_data.head(200), label='Clean', color='blue')
    plt.title(f'Original Data')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 2)
    plt.plot(trends['clean'].head(200), label='Clean', color='blue')
    plt.title(f'Trend')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 3)
    plt.plot(seasonals['clean'].head(200), label='Clean', color='blue')
    plt.title(f'Seasonal')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 4)
    plt.plot(resids['clean'].head(200), label='Clean', color='blue')
    plt.title(f'Residuals')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_path, f'decompose_basic_clean.png'))


    for model in Imputation_Algorithms:
        file_50 = f'dirty-{model}-50.csv'
        data_50 = dfs_dirty[file_50][target_column]
        decomp_50 = decompositions[file_50]
        trend_50 = decomp_50.trend.dropna()
        seasonal_50 = decomp_50.seasonal.dropna()
        resid_50 = decomp_50.resid.dropna()

        plt.figure(figsize=(14, 12))

        plt.subplot(4, 1, 1)
        plt.plot(clean_data.head(200), label='Clean', color='blue')
        plt.plot(data_50.head(200), label=model, color='red')
        plt.title(f'Original Data - {model} 50%')
        plt.legend(loc='upper left')

        plt.subplot(4, 1, 2)
        plt.plot(trends['clean'].head(200), label='Clean', color='blue')
        plt.plot(trend_50.head(200), label=model, color='red')
        plt.title(f'Trend - {model} 50%')
        plt.legend(loc='upper left')

        plt.subplot(4, 1, 3)
        plt.plot(seasonals['clean'].head(200), label='Clean', color='blue')
        plt.plot(seasonal_50.head(200), label=model, color='red')
        plt.title(f'Seasonal - {model} 50%')
        plt.legend(loc='upper left')

        plt.subplot(4, 1, 4)
        plt.plot(resids['clean'].head(200), label='Clean', color='blue')
        plt.plot(resid_50.head(200), label=model, color='red')
        plt.title(f'Residuals - {model} 50%')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, f'decompose_basic_{model}_50.png'))
        #plt.show()