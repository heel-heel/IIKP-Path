import pandas as pd
import os
from statsmodels.tsa.seasonal import seasonal_decompose


cycle = 12
missing_rate_to_model = {
    50: 'si',
    70: 'randomforest',
    90: 'si'
}
models_original = ['mean', 'median', 'mfi', 'gan', 'midae']

for missing_rate, model_trend in missing_rate_to_model.items():
    for model_original in models_original:
        save_path = f'./null-trend_{model_original}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        trend_file = pd.read_csv(f'./null-{model_trend}/dirty-{model_trend}-{missing_rate}.csv')
        original_file = pd.read_csv(f'./null-{model_original}/dirty-{model_original}-{missing_rate}.csv')
        dirty_file = pd.read_csv(f'./null/dirty-{missing_rate}.csv')
        output_file = os.path.join(save_path, f'dirty-trend_{model_original}-{missing_rate}.csv')

        trend_decomposed = seasonal_decompose(trend_file['V2'],model='additive',period=cycle)
        seasonal_residuals_decomposed = seasonal_decompose(original_file['V2'],model='additive',period=cycle)

        trend = trend_decomposed.trend
        seasonal = seasonal_residuals_decomposed.seasonal
        residuals = seasonal_residuals_decomposed.resid

        new_data = residuals.add(trend).add(seasonal)
        half_cycle = cycle // 2
        new_data[:half_cycle] = trend_file['V2'][:half_cycle]
        new_data[-half_cycle:] = trend_file['V2'][-half_cycle:]

        dirty_file_filled = dirty_file.copy()
        dirty_file_filled['V2'] = dirty_file_filled['V2'].fillna(new_data)
        dirty_file_filled.to_csv(os.path.join(save_path,f'dirty-trend_{model_original}-{missing_rate}.csv'),index=False)
        print(f'"dirty-trend_{model_original}-{missing_rate}.csv"已保存到{save_path}.')