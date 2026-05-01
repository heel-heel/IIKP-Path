import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

cycle = 12
input_clean_file = os.path.join("generate_timeseries_results", "synthetic_time_series_balanced.csv")
clean_df = pd.read_csv(input_clean_file)
clean_data = clean_df["y_combined"]

decompositions = {'clean': seasonal_decompose(clean_data, model="additive", period=cycle)}
trends = {name: decomp.trend.dropna() for name, decomp in decompositions.items()}
seasonals = {name: decomp.seasonal.dropna() for name, decomp in decompositions.items()}
resids = {name: decomp.resid.dropna() for name, decomp in decompositions.items()}

plt.figure(figsize=(14, 12))

plt.subplot(4, 1, 1)
plt.plot(clean_data, label='Clean', color='blue')
plt.title(f'Original Data')
plt.legend(loc='upper left')

plt.subplot(4, 1, 2)
plt.plot(trends['clean'], label='Clean', color='blue')
plt.title(f'Trend')
plt.legend(loc='upper left')

plt.subplot(4, 1, 3)
plt.plot(seasonals['clean'], label='Clean', color='blue')
plt.title(f'Seasonal')
plt.legend(loc='upper left')

plt.subplot(4, 1, 4)
plt.plot(resids['clean'], label='Clean', color='blue')
plt.title(f'Residuals')
plt.legend(loc='upper left')

plt.show()

