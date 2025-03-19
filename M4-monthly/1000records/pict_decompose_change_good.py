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

save_path='./decomposition-resid-50/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_pic_path='./decomposition-resid-50/pic/'
if not os.path.exists(save_pic_path):
    os.makedirs(save_pic_path)


# 定义函数生成dirty_trend并计算相关系数
def generate_dirty_trend_and_corr(std_dev, fraction=0.5):
    np.random.seed(0)
    n = len(df['resid'])
    indices = np.random.choice(n, size=int(n * fraction), replace=False)  # 随机选择部分数据点
    intercept = np.random.normal(0, std_dev)  # 添加随机噪声，控制相关系数
    dirty_resid = df['resid'].copy()
    dirty_resid.iloc[indices] += intercept + np.random.normal(0, std_dev, size=len(indices))
    dirty_data = dirty_resid + df['trend'] + df['seasonal']
    dirty_df = pd.DataFrame({'V2': dirty_data, 'trend': df['trend'], 'seasonal': df['seasonal'], 'resid': dirty_resid})

    dirty_df.iloc[:6, 0] = df.iloc[:6, 1]
    dirty_df.iloc[-6:, 0] = df.iloc[-6:, 1]
    dirty_df['resid'] = dirty_df['resid'].fillna(0)
    corr_trend = np.corrcoef(df['resid'], dirty_df['resid'])[0, 1]
    return corr_trend, dirty_df

# 目标相关系数列表
target_corrs = [0.70,0.72,0.74,0.76,0.78,0.80,0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]

# 逐步尝试不同的标准差
std_devs = np.linspace(1000, 6000.0, 100000)

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
        corr_trend2 = np.corrcoef(data, dirty_df['V2'])[0, 1]

        print("--------------------------------")
        print(f"目标相关系数：{target_corr}")
        print(f"最佳标准差：{best_std_dev}")
        print(f"最佳相关系数：{best_corr}")
        print(f"修改后的数据与clean的相关系数：{corr_trend2}")

        # 创建一个包含新结果的DataFrame
        new_row = pd.DataFrame({
            'Target Correlation': [target_corr],
            'Best Standard Deviation': [best_std_dev],
            'Best Correlation': [best_corr],
            'Original vs Generated Correlation': [corr_trend2]
        })

        # 使用pd.concat()将新行添加到结果DataFrame中
        results = pd.concat([results, new_row], ignore_index=True)

        # 导出dirty数据集为CSV文件
        df_copy['V2'] = dirty_df['V2']
        df_copy.to_csv(os.path.join(save_path,f'dirty-resid-50-{int(target_corr * 100)}.csv'), index=False)

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
        plt.savefig(os.path.join(save_pic_path,f'decomposition_resid_200_{int(target_corr * 100)}.png'))  # 保存图像
        plt.close()

# 导出结果到CSV文件
results.to_csv(os.path.join(save_path,'decomposition_resid_results.csv'), index=False)