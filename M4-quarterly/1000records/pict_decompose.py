import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 定义模型名称和百分比
model_names = ['mean','median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'randomforest', 'xgboost', 'gan', 'midae']
percentages = [10, 30, 50, 70, 90]
csv_files = [f'dirty-{model}-{p}.csv' for model in model_names for p in percentages]

save_path='./decomposition-200/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_pic_path='./decomposition-200/pic/'
if not os.path.exists(save_pic_path):
    os.makedirs(save_pic_path)

# 读取clean数据
df_clean = pd.read_csv("clean.csv")
df_clean = df_clean.head(200)
#df_clean = df_clean
data_clean = df_clean["V2"]

# 读取所有dirty数据
dfs_dirty = {file: pd.read_csv(os.path.join(f"null-{file.split('-')[1]}",file)).head(200) for file in csv_files}
#dfs_dirty = {file: pd.read_csv(file) for file in csv_files}
data_dirty = {file: df["V2"] for file, df in dfs_dirty.items()}

# 时序分解
decompositions = {'clean': seasonal_decompose(data_clean, model="additive", period=12)}
decompositions.update({file: seasonal_decompose(data, model="additive", period=12) for file, data in data_dirty.items()})

# 提取趋势、季节性和残差成分
trends = {name: decomp.trend.dropna() for name, decomp in decompositions.items()}
seasonals = {name: decomp.seasonal.dropna() for name, decomp in decompositions.items()}
resids = {name: decomp.resid.dropna() for name, decomp in decompositions.items()}

# 计算指标
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
        'Model': file,
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

# 保存结果到CSV文件
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_path,"decomposition_200_results.csv"), index=False)

# 打印结果
print("Results saved to 'decomposition_results.csv'")

# 绘制50%情况下的图像
for model in model_names:
    file_50 = f'dirty-{model}-50.csv'
    data_50 = dfs_dirty[file_50]["V2"]
    decomp_50 = decompositions[file_50]
    trend_50 = decomp_50.trend.dropna()
    seasonal_50 = decomp_50.seasonal.dropna()
    resid_50 = decomp_50.resid.dropna()

    plt.figure(figsize=(14, 12))

    # 绘制原始数据
    plt.subplot(4, 1, 1)
    plt.plot(data_clean, label='Clean', color='blue')
    plt.plot(data_50, label=model, color='red')
    plt.title(f'Original Data - {model} 50%')
    plt.legend(loc='upper left')

    # 绘制趋势
    plt.subplot(4, 1, 2)
    plt.plot(trends['clean'], label='Clean', color='blue')
    plt.plot(trend_50, label=model, color='red')
    plt.title(f'Trend - {model} 50%')
    plt.legend(loc='upper left')

    # 绘制季节性
    plt.subplot(4, 1, 3)
    plt.plot(seasonals['clean'], label='Clean', color='blue')
    plt.plot(seasonal_50, label=model, color='red')
    plt.title(f'Seasonal - {model} 50%')
    plt.legend(loc='upper left')

    # 绘制残差
    plt.subplot(4, 1, 4)
    plt.plot(resids['clean'], label='Clean', color='blue')
    plt.plot(resid_50, label=model, color='red')
    plt.title(f'Residuals - {model} 50%')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(save_pic_path,f'decomposition_200_{model}_50.png'))
    plt.show()