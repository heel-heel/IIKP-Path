import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('clean.csv', parse_dates=['Month'])

# 找到峰值和谷值
peaks = []
valleys = []

for i in range(1, len(df) - 1):
    if df.iloc[i]['Sunspots'] > df.iloc[i - 1]['Sunspots'] and df.iloc[i]['Sunspots'] > df.iloc[i + 1]['Sunspots']:
        peaks.append(i)
    if df.iloc[i]['Sunspots'] < df.iloc[i - 1]['Sunspots'] and df.iloc[i]['Sunspots'] < df.iloc[i + 1]['Sunspots']:
        valleys.append(i)

# 将峰值和谷值的10%的数据变为空值
num_peaks_to_nan = max(int(len(peaks) *0.9), 1)  # 至少一个峰值变为空值
num_valleys_to_nan = max(int(len(valleys)*0.9) , 1)  # 至少一个谷值变为空值
print((len(peaks)+len(valleys))/len(df))
print((num_peaks_to_nan+num_valleys_to_nan)/len(df))

peaks_to_nan = np.random.choice(peaks, size=num_peaks_to_nan, replace=False)
valleys_to_nan = np.random.choice(valleys, size=num_valleys_to_nan, replace=False)

df.loc[peaks_to_nan, 'Sunspots'] = np.nan
df.loc[valleys_to_nan, 'Sunspots'] = np.nan

# 导出修改后的数据集
#df.to_csv('dirty-pv-90.csv', index=False)