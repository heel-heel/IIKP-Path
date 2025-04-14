import pandas as pd
import os
from statsmodels.tsa.seasonal import seasonal_decompose

datasets = {
    "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"}
}
missing_rate_to_model = {
    "M4-Monthly": {
        50: 'si',
        70: 'si',
        90: 'si'
    },
    "M4-Quarterly": {
        50: 'si',
        70: 'rf',
        90: 'si'
    },
    "M4-Yearly": {
        50: 'si',
        70: 'si',
        90: 'si'
    }
}
models_original = ['mean', 'median', 'mfi', 'gain', 'midae']
cycle = 12

for dataset, columns in datasets.items():
    current_missing_rate_to_model = missing_rate_to_model[dataset]
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]
    base_path = "../../../Datasets"
    output_path = os.path.join(base_path, dataset, "Mechanism", "timeseries", "decompose_replace", "resid")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for rate, model_resid in current_missing_rate_to_model.items():
        for model_original in models_original:
            input_resid_file = os.path.join(base_path, dataset, "Imputation", f"null-{model_resid}", f"dirty-{model_resid}-{rate}.csv")
            input_original_file = os.path.join(base_path, dataset, "Imputation", f"null-{model_original}", f"dirty-{model_original}-{rate}.csv")
            input_dirty_file = os.path.join(base_path, dataset, "null", f"dirty-{rate}")

            residuals_file = pd.read_csv(input_resid_file)
            original_file = pd.read_csv(input_original_file)
            dirty_file = pd.read_csv(input_dirty_file)

            residuals_decomposed = seasonal_decompose(residuals_file[target_column], model='additive', period=cycle)
            trend_seasonal_decomposed = seasonal_decompose(original_file[target_column], model='additive', period=cycle)

            residuals = residuals_decomposed.resid
            trend = trend_seasonal_decomposed.trend
            seasonal = trend_seasonal_decomposed.seasonal

            new_data = residuals.add(trend).add(seasonal)
            half_cycle = cycle // 2
            new_data[:half_cycle] = residuals_file[target_column][:half_cycle]
            new_data[-half_cycle:] = residuals_file[target_column][-half_cycle:]

            dirty_file_filled = dirty_file.copy()
            dirty_file_filled[target_column] = dirty_file_filled[target_column].fillna(new_data)

            output_file = os.path.join(output_path, f'dirty-resid_{model_original}-{rate}.csv')
            dirty_file_filled.to_csv(output_file, index=False)
            print(f'"dirty-resid_{model_original}-{rate}.csv" has saved to {output_path}.')