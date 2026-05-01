# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# 配置参数
datasets = ['ETTh1', 'ETTm1', 'Illness', 'Exchange', 'Weather']
models = ['mlp', 'tsmixer', 'lightts', 'timemixer']
pattern = 'MCAR'
missing_rates = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

base_path = "../../Downstream_Results/timeseries"

# 存储每个数据集下各模型与MLP的相关系数
results = {}


def extract_pg_values(file_path, model_name):
    """
    从CSV文件中提取指定模型在各缺失率下的PG(RMSE)值
    """
    try:
        df = pd.read_csv(file_path)
        pg_values = []

        for rate in missing_rates:
            target_pattern = f"dirty-{rate}.csv"
            matching_rows = df[df['File Name'] == target_pattern]
            if not matching_rows.empty:
                pg_value = matching_rows['PG(RMSE)'].values[0]
                pg_values.append(pg_value)
            else:
                print(f"  Warning: {target_pattern} not found in {file_path}")
                pg_values.append(np.nan)

        return pg_values
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return None


# 遍历每个数据集
for dataset in datasets:
    print(f"\n处理数据集: {dataset}")
    results[dataset] = {}

    # 存储各模型的PG值
    model_pg_values = {}

    # 提取每个模型的PG值
    for model in models:
        file_path = os.path.join(base_path, dataset, pattern, f"{model}-imputation-results-{dataset}.csv")

        if os.path.exists(file_path):
            pg_values = extract_pg_values(file_path, model)
            if pg_values:
                model_pg_values[model] = pg_values
                print(f"  {model}: PG values extracted")
            else:
                print(f"  {model}: Failed to extract PG values")
        else:
            print(f"  Warning: File not found - {file_path}")

    # 计算相关系数（MLP vs 其他模型）
    if 'mlp' in model_pg_values:
        mlp_values = model_pg_values['mlp']

        for other_model in ['tsmixer', 'lightts', 'timemixer']:
            if other_model in model_pg_values:
                other_values = model_pg_values[other_model]

                # 移除缺失值
                valid_mask = ~(np.isnan(mlp_values) | np.isnan(other_values))
                mlp_clean = np.array(mlp_values)[valid_mask]
                other_clean = np.array(other_values)[valid_mask]

                if len(mlp_clean) > 2:
                    corr, p_value = pearsonr(mlp_clean, other_clean)
                    results[dataset][other_model] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'n': len(mlp_clean)
                    }
                    print(f"  {other_model} vs MLP: ρ = {corr:.4f}, p = {p_value:.4f}, n = {len(mlp_clean)}")
                else:
                    results[dataset][other_model] = None
                    print(f"  {other_model} vs MLP: Insufficient data")
            else:
                results[dataset][other_model] = None
                print(f"  {other_model}: No data available")
    else:
        print(f"  MLP data not available for {dataset}")

# 输出汇总结果
print("\n" + "=" * 80)
print("汇总结果：各数据集下深度模型与MLP的相关系数")
print("=" * 80)
print(f"{'数据集':<12} {'TSMixer vs MLP':<20} {'LightTS vs MLP':<20} {'TimeMixer vs MLP':<20}")
print("-" * 72)

for dataset in datasets:
    if dataset in results:
        tsmixer_val = results[dataset].get('tsmixer')
        lightts_val = results[dataset].get('lightts')
        timemixer_val = results[dataset].get('timemixer')

        tsmixer_str = f"{tsmixer_val['correlation']:.4f}" if tsmixer_val and tsmixer_val[
            'correlation'] is not None else "N/A"
        lightts_str = f"{lightts_val['correlation']:.4f}" if lightts_val and lightts_val[
            'correlation'] is not None else "N/A"
        timemixer_str = f"{timemixer_val['correlation']:.4f}" if timemixer_val and timemixer_val[
            'correlation'] is not None else "N/A"

        print(f"{dataset:<12} {tsmixer_str:<20} {lightts_str:<20} {timemixer_str:<20}")
    else:
        print(f"{dataset:<12} {'N/A':<20} {'N/A':<20} {'N/A':<20}")

# 计算平均相关系数
print("\n" + "=" * 80)
print("平均相关系数（跨所有数据集）")
print("=" * 80)

avg_tsmixer = np.mean([results[d]['tsmixer']['correlation'] for d in datasets if
                       d in results and results[d].get('tsmixer') and results[d]['tsmixer']['correlation'] is not None])
avg_lightts = np.mean([results[d]['lightts']['correlation'] for d in datasets if
                       d in results and results[d].get('lightts') and results[d]['lightts']['correlation'] is not None])
avg_timemixer = np.mean([results[d]['timemixer']['correlation'] for d in datasets if
                         d in results and results[d].get('timemixer') and results[d]['timemixer'][
                             'correlation'] is not None])

print(f"TSMixer vs MLP 平均相关系数: {avg_tsmixer:.4f}")
print(f"LightTS vs MLP 平均相关系数: {avg_lightts:.4f}")
print(f"TimeMixer vs MLP 平均相关系数: {avg_timemixer:.4f}")

# 保存结果到文件
output_file = os.path.join("results", "model_correlation_analysis_mlp.txt")
os.makedirs("results", exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("深度模型与MLP在PG(RMSE)上的相关系数分析\n")
    f.write(f"缺失模式: {pattern}\n")
    f.write("=" * 80 + "\n\n")

    for dataset in datasets:
        if dataset in results:
            f.write(f"\n数据集: {dataset}\n")
            f.write(f"{'模型对比':<20} {'相关系数 ρ':<15} {'p-value':<15} {'样本数 n':<10}\n")
            f.write("-" * 60 + "\n")

            for other_model in ['tsmixer', 'lightts', 'timemixer']:
                if results[dataset].get(other_model):
                    val = results[dataset][other_model]
                    model_name = other_model.upper()
                    f.write(
                        f"MLP vs {model_name:<13} {val['correlation']:<15.4f} {val['p_value']:<15.4f} {val['n']:<10}\n")
                else:
                    f.write(f"MLP vs {other_model.upper():<13} {'N/A':<15} {'N/A':<15} {'N/A':<10}\n")

    f.write(f"\n{'=' * 80}\n")
    f.write("平均相关系数（跨所有数据集）\n")
    f.write(f"TSMixer vs MLP: {avg_tsmixer:.4f}\n")
    f.write(f"LightTS vs MLP: {avg_lightts:.4f}\n")
    f.write(f"TimeMixer vs MLP: {avg_timemixer:.4f}\n")
    f.write("=" * 80 + "\n")

print(f"\n结果已保存到: {output_file}")