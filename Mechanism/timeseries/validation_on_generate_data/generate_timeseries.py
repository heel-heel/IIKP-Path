# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_balanced_time_series(
        n_points=1000,
        trend_slope=1.0,
        seasonal_amplitude=1.0,
        seasonal_period=12,
        noise_scale=1.0,
        random_seed=42
):
    """
    生成三个成分强度相等的合成时间序列（加法模型）

    Parameters:
    -----------
    n_points : int
        时间序列长度
    trend_slope : float
        线性趋势的斜率
    seasonal_amplitude : float
        季节性成分的振幅
    seasonal_period : int
        季节性周期（如12表示年度周期）
    noise_scale : float
        高斯噪声的标准差
    random_seed : int
        随机种子

    Returns:
    --------
    y : np.ndarray
        合成时间序列
    T : np.ndarray
        趋势成分
    S : np.ndarray
        季节性成分
    R : np.ndarray
        残差/噪声成分
    """
    np.random.seed(random_seed)

    t = np.arange(n_points)

    # 1. 线性趋势成分 T(t) = slope * t
    T = trend_slope * t

    # 2. 单正弦波季节性成分 S(t) = amplitude * sin(2π * t / period)
    S = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)

    # 3. 高斯噪声残差 R(t) ~ N(0, scale^2)
    R = np.random.normal(0, noise_scale, n_points)

    # 4. 加法模型：y(t) = T(t) + S(t) + R(t)
    y = T + S + R

    return y, T, S, R


def standardize_components_to_equal_strength(
        n_points=1000,
        seasonal_period=12,
        target_std=1.0,
        random_seed=42
):
    """
    生成三个成分强度相等（标准化后标准差均为 target_std）的时间序列

    通过先分别生成各成分，再标准化到相同的标准差，确保三者强度相等

    Returns:
    --------
    y : np.ndarray
        合成时间序列
    T_scaled : np.ndarray
        标准化后的趋势成分（标准差 = target_std）
    S_scaled : np.ndarray
        标准化后的季节性成分（标准差 = target_std）
    R_scaled : np.ndarray
        标准化后的残差成分（标准差 = target_std）
    """
    np.random.seed(random_seed)

    t = np.arange(n_points)

    # 1. 线性趋势（范围从0到n_points-1）
    T_raw = t

    # 2. 单正弦波季节性（振幅=1，周期=seasonal_period）
    S_raw = np.sin(2 * np.pi * t / seasonal_period)

    # 3. 高斯噪声（标准差=1）
    R_raw = np.random.normal(0, 1, n_points)

    # 标准化各成分到均值为0，标准差为target_std
    T_scaled = (T_raw - T_raw.mean()) / T_raw.std() * target_std
    S_scaled = (S_raw - S_raw.mean()) / S_raw.std() * target_std
    R_scaled = (R_raw - R_raw.mean()) / R_raw.std() * target_std

    # 加法合成
    y = T_scaled + S_scaled + R_scaled

    return y, T_scaled, S_scaled, R_scaled


# ============ 生成并导出数据 ============

# 参数设置
N_POINTS = 1000
SEASONAL_PERIOD = 12
TARGET_STD = 1.0
RANDOM_SEED = 42

# 生成强度相等的时间序列
y, T, S, R = standardize_components_to_equal_strength(
    n_points=N_POINTS,
    seasonal_period=SEASONAL_PERIOD,
    target_std=TARGET_STD,
    random_seed=RANDOM_SEED
)

# 验证各成分标准差是否相等
print("=" * 50)
print("成分强度验证（标准差）")
print("=" * 50)
print(f"趋势成分 (Trend) 标准差: {np.std(T):.6f}")
print(f"季节性成分 (Seasonality) 标准差: {np.std(S):.6f}")
print(f"残差成分 (Residuals) 标准差: {np.std(R):.6f}")
print(f"目标标准差: {TARGET_STD}")
print()

# 验证均值为0
print("成分均值验证")
print("=" * 50)
print(f"趋势成分 (Trend) 均值: {np.mean(T):.6f}")
print(f"季节性成分 (Seasonality) 均值: {np.mean(S):.6f}")
print(f"残差成分 (Residuals) 均值: {np.mean(R):.6f}")
print()

# ============ 导出CSV文件 ============

# 创建DataFrame
df = pd.DataFrame({
    'time_index': np.arange(N_POINTS),
    'trend': T,
    'seasonality': S,
    'residuals': R,
    'y_combined': y
})

# 保存为CSV
csv_filename = 'generate_timeseries_results/synthetic_time_series_balanced.csv'
df.to_csv(csv_filename, index=False)
print(f"时间序列已导出至: {csv_filename}")
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")

# ============ 可视化 ============

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 各成分单独展示（前200个点）
axes[0, 0].plot(T[:200], color='red', linewidth=1.5)
axes[0, 0].set_title('Trend Component (Linear)', fontsize=12)
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(S[:200], color='green', linewidth=1.5)
axes[0, 1].set_title('Seasonality Component (Sine Wave)', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(R[:200], color='blue', linewidth=1.5)
axes[1, 0].set_title('Residuals Component (Gaussian Noise)', fontsize=12)
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Value')
axes[1, 0].grid(True, alpha=0.3)

# 合成时间序列
axes[1, 1].plot(y[:500], color='black', linewidth=1)
axes[1, 1].set_title('Synthetic Time Series (T + S + R)', fontsize=12)
axes[1, 1].set_xlabel('Time')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('synthetic_time_series_balanced.png', dpi=150, bbox_inches='tight')
plt.show()

# ============ 打印数据样本 ============
print("\n数据样本（前10行）:")
print("=" * 60)
print(df.head(10).to_string(index=False))