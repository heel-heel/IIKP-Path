import pandas as pd
import os
from statsmodels.tsa.seasonal import seasonal_decompose


cycle = 12
missing_rate_to_model = {
    50: 'si',
    70: 'si',
    90: 'si'
}
models_original = ['mean', 'median', 'mfi', 'gan', 'midae']

for missing_rate, model_seasonal in missing_rate_to_model.items():
    for model_original in models_original:
        save_path = f'./null-seasonal_{model_original}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        seasonal_file = pd.read_csv(f'./null-{model_seasonal}/dirty-{model_seasonal}-{missing_rate}.csv')
        original_file = pd.read_csv(f'./null-{model_original}/dirty-{model_original}-{missing_rate}.csv')
        dirty_file = pd.read_csv(f'./null/dirty-{missing_rate}.csv')
        output_file = os.path.join(save_path, f'dirty-seasonal_{model_original}-{missing_rate}.csv')

        seasonal_decomposed = seasonal_decompose(seasonal_file['V2'],model='additive',period=cycle)
        residuals_trend_decomposed = seasonal_decompose(original_file['V2'],model='additive',period=cycle)

        seasonal = seasonal_decomposed.seasonal
        residuals = residuals_trend_decomposed.resid
        trend = residuals_trend_decomposed.trend

        new_data = residuals.add(trend).add(seasonal)
        half_cycle = cycle // 2
        new_data[:half_cycle] = seasonal_file['V2'][:half_cycle]
        new_data[-half_cycle:] = seasonal_file['V2'][-half_cycle:]

        dirty_file_filled = dirty_file.copy()
        dirty_file_filled['V2'] = dirty_file_filled['V2'].fillna(new_data)
        dirty_file_filled.to_csv(os.path.join(save_path,f'dirty-seasonal_{model_original}-{missing_rate}.csv'),index=False)
        print(f'"dirty-seasonal_{model_original}-{missing_rate}.csv"已保存到{save_path}.')