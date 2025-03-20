import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 读取数据
plt.rc("figure", figsize=(10, 6))
df = pd.read_csv("clean.csv")
df_copy = pd.read_csv("clean.csv")
data = df["V2"]
seasonal_decomp = seasonal_decompose(data, model="additive", period=12)
df['trend'] = seasonal_decomp.trend
df['seasonal'] = seasonal_decomp.seasonal
df['resid'] = seasonal_decomp.resid

# 填充开头和末尾的空值
df['trend'] = df['trend'].fillna(0)
df['seasonal'] = df['seasonal'].fillna(0)
df['resid'] = df['resid'].fillna(0)

save_path='./decomposition-bad/decomposition-trend-50/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_pic_path='./decomposition-bad/decomposition-trend-50/pic/'
if not os.path.exists(save_pic_path):
    os.makedirs(save_pic_path)

# 定义函数生成带有噪声的trend和resid
def add_noise_to_seasonal_and_resid(target_corr_seasonal, target_corr_resid, fraction=0.5):
    np.random.seed(0)
    n = len(df['seasonal'])
    indices = np.random.choice(n, size=int(n * fraction), replace=False)  # 随机选择部分数据点

    # 添加噪声到trend
    best_std_dev_seasonal = None
    best_corr_seasonal = None
    min_diff_seasonal = float('inf')
    std_devs_seasonal = np.linspace(100, 6000.0, 10000)  # 调整范围以适应数据

    for std_dev in std_devs_seasonal:
        noisy_seasonal = df['seasonal'].copy()
        noisy_seasonal.iloc[indices] += np.random.normal(0, std_dev, size=len(indices))
        corr_seasonal = np.corrcoef(df['seasonal'], noisy_seasonal)[0, 1]
        diff = abs(corr_seasonal - target_corr_seasonal)
        if diff < min_diff_seasonal:
            min_diff_seasonal = diff
            best_std_dev_seasonal = std_dev
            best_corr_seasonal = corr_seasonal
        if min_diff_seasonal < 1e-5:
            break

    # 添加噪声到resid
    best_std_dev_resid = None
    best_corr_resid = None
    min_diff_resid = float('inf')
    std_devs_resid = np.linspace(100, 10000.0, 100000)  # 调整范围以适应数据

    for std_dev in std_devs_resid:
        noisy_resid = df['resid'].copy()
        noisy_resid.iloc[indices] += np.random.normal(0, std_dev, size=len(indices))
        corr_resid = np.corrcoef(df['resid'], noisy_resid)[0, 1]
        diff = abs(corr_resid - target_corr_resid)
        if diff < min_diff_resid:
            min_diff_resid = diff
            best_std_dev_resid = std_dev
            best_corr_resid = corr_resid
        if min_diff_seasonal < 1e-5:
            break

    return best_std_dev_seasonal, best_corr_seasonal, noisy_seasonal, best_std_dev_resid, best_corr_resid, noisy_resid

# 添加噪声到trend和seasonal
target_corr_seasonal = 0.5
target_corr_resid = 0.5
best_std_dev_seasonal, best_corr_seasonal, noisy_seasonal, best_std_dev_resid, best_corr_resid, noisy_resid = add_noise_to_seasonal_and_resid(target_corr_seasonal, target_corr_resid)

print(f"最佳seasonal标准差：{best_std_dev_seasonal}, 最佳seasonal相关系数：{best_corr_seasonal}")
print(f"最佳resid标准差：{best_std_dev_resid}, 最佳resid相关系数：{best_corr_resid}")

# 定义函数生成dirty_resid并计算相关系数
def generate_dirty_trend_and_corr(std_dev, fraction=0.5):
    np.random.seed(0)
    n = len(df['resid'])
    indices = np.random.choice(n, size=int(n * fraction), replace=False)  # 随机选择部分数据点
    intercept = np.random.normal(0, std_dev)  # 添加随机噪声，控制相关系数
    dirty_trend = df['trend'].copy()
    dirty_trend.iloc[indices] += intercept + np.random.normal(0, std_dev, size=len(indices))
    dirty_data = dirty_trend + noisy_seasonal + noisy_resid
    dirty_df = pd.DataFrame({'V2': dirty_data, 'trend': dirty_trend, 'seasonal': noisy_seasonal, 'resid': noisy_resid})

    dirty_df.iloc[:6, 0] = df.iloc[:6, 1]
    dirty_df.iloc[-6:, 0] = df.iloc[-6:, 1]
    dirty_df['trend'] = dirty_df['trend'].fillna(0)
    corr_resid = np.corrcoef(df['trend'], dirty_df['trend'])[0, 1]
    return corr_resid, dirty_df

# 目标相关系数列表
target_corrs = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]

# 逐步尝试不同的标准差
std_devs = np.linspace(1, 6000.0, 100000)

# 创建一个DataFrame来存储结果
results = pd.DataFrame(columns=['Target Correlation', 'Best Standard Deviation', 'Best Correlation', 'Original vs Generated Correlation'])

# 遍历每个目标相关系数
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
        corr_clean_dirty = np.corrcoef(data, dirty_df['V2'])[0, 1]

        print("--------------------------------")
        print(f"目标相关系数：{target_corr}")
        print(f"最佳标准差：{best_std_dev}")
        print(f"最佳相关系数：{best_corr}")
        print(f"修改后的数据与clean的相关系数：{corr_clean_dirty}")

        # 创建一个包含新结果的DataFrame
        new_row = pd.DataFrame({
            'Target Correlation': [target_corr],
            'Best Standard Deviation': [best_std_dev],
            'Best Correlation': [best_corr],
            'Original vs Generated Correlation': [corr_clean_dirty]
        })

        # 使用pd.concat()将新行添加到结果DataFrame中
        results = pd.concat([results, new_row], ignore_index=True)

        # 导出dirty数据集为CSV文件
        df_copy['V2'] = dirty_df['V2']
        df_copy.to_csv(os.path.join(save_path, f'dirty-trend-50-{int(target_corr * 100)}.csv'), index=False)

        # 绘制图像
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        df_head = df.head(200)
        dirty_df_head = dirty_df.head(200)

        axs[0].plot(df_head.index, df_head['V2'], label='Clean', color='blue')
        axs[0].plot(dirty_df_head.index, dirty_df_head['V2'], label='Dirty', color='red')
        axs[0].set_title('Original Data')
        axs[0].legend()

        # Trend
        axs[1].plot(df_head.index, df_head['trend'], label='Clean', color='blue')
        axs[1].plot(dirty_df_head.index, dirty_df_head['trend'], label='Dirty', color='red')
        axs[1].set_title('Trend')
        axs[1].legend()

        # Seasonal
        axs[2].plot(df_head.index, df_head['seasonal'], label='Clean', color='blue')
        axs[2].plot(dirty_df_head.index, dirty_df_head['seasonal'], label='Dirty', color='red')
        axs[2].set_title('Seasonal')
        axs[2].legend()

        # Resid
        axs[3].plot(df_head.index, df_head['resid'], label='Clean', color='blue')
        axs[3].plot(dirty_df_head.index, dirty_df_head['resid'], label='Dirty', color='red')
        axs[3].set_title('Residual')
        axs[3].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_pic_path, f'decomposition_trend_200_{int(target_corr * 100)}.png'))  # 保存图像
        plt.close()

# 导出结果到CSV文件
results.to_csv(os.path.join(save_path, 'decomposition_trend_results.csv'), index=False)