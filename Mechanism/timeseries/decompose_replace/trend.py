import pandas as pd
import os
from statsmodels.tsa.seasonal import seasonal_decompose

datasets = {
    #"ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},#24
    #"ETTm1": {"target_column": "OT", "nonnumerical_column": "date"},#96
    #"Illness": {"target_column": "OT", "nonnumerical_column": "date"},#52
    #"Exchange": {"target_column": "OT", "nonnumerical_column": "date"},#7
    "Weather": {"target_column": "OT", "nonnumerical_column": "date"},#6
}
Missing_Mechanism = 'MCAR'
cycle = 6
missing_rate_to_model = {
    "ETTh1": {
        50: 'midae',
        70: 'midae',
        90: 'midae'
    },
    "ETTm1": {
        50: 'si',
        70: 'si',
        90: 'midae'
    },
    "Illness": {
        50: 'xgbi',
        70: 'xgbi',
        90: 'xgbi'
    },
    "Exchange": {
        50: 'hdi',
        70: 'hdi',
        90: 'hdi'
    },
    "Weather": {
        50: 'xgbi',
        70: 'xgbi',
        90: 'xgbi'
    },

}
models_original = ['mean', 'median', 'mode', 'si', 'mfi', 'gain', 'midae']


for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]
    current_missing_rate_to_model = missing_rate_to_model[dataset]

    base_path = "../../../Datasets"
    output_path = os.path.join(base_path, dataset, "Mechanism", "timeseries", "decompose_replace", "trend")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for rate, model_trend in current_missing_rate_to_model.items():
        for model_original in models_original:
            input_trend_file = os.path.join(base_path, dataset, "Imputation", Missing_Mechanism, f"null-{model_trend}", f"dirty-{model_trend}-{rate}.csv")
            input_original_file = os.path.join(base_path, dataset, "Imputation", Missing_Mechanism, f"null-{model_original}", f"dirty-{model_original}-{rate}.csv")
            input_dirty_file = os.path.join(base_path, dataset, "null", Missing_Mechanism, f"dirty-{rate}.csv")

            trend_file = pd.read_csv(input_trend_file)
            original_file = pd.read_csv(input_original_file)
            dirty_file = pd.read_csv(input_dirty_file)

            trend_decomposed = seasonal_decompose(trend_file[target_column], model='additive', period=cycle)
            seasonal_residuals_decomposed = seasonal_decompose(original_file[target_column], model='additive', period=cycle)

            trend = trend_decomposed.trend
            seasonal = seasonal_residuals_decomposed.seasonal
            residuals = seasonal_residuals_decomposed.resid

            new_data = residuals.add(trend).add(seasonal)
            half_cycle = cycle // 2
            new_data[:half_cycle] = trend_file[target_column][:half_cycle]
            new_data[-half_cycle:] = trend_file[target_column][-half_cycle:]

            dirty_file_filled = dirty_file.copy()
            dirty_file_filled[target_column] = dirty_file_filled[target_column].fillna(new_data)

            # 保存结果
            output_file = os.path.join(output_path, f'dirty-trend_{model_original}-{rate}.csv')
            dirty_file_filled.to_csv(output_file, index=False)
            print(f'"dirty-trend_{model_original}-{rate}.csv" has saved to {output_path}.')