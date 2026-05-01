import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')

# 配置参数
tasks = {
    'timeseries': {
        'datasets': ['ETTh1', 'ETTm1', 'Illness', 'Exchange', 'Weather'],
        'models': ['mlp'],
    },
    'regression': {
        'datasets': ['concrete', 'CCPP', 'AirfoilSelfNoise', 'Abalone', 'ParisHousing'],
        'models': ['mlp']
    }
}

Imputation_Algorithms = ['Mean', 'Median', 'Mode', 'KNN', 'HDI', 'MICE', 'IIM', 'SI',
                         'MFI', 'MissFI', 'XGBI', 'GAIN', 'MIDAE']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
task_metrics = {
    'timeseries': {'metric': 'PG(RMSE)'},
    'regression': {'metric': 'PG(MAE)'}
}
Mechanism = ['MCAR', 'MAR', 'MNAR']

# 数据质量指标
quality_metrics = ['2_wasserstein_distance', 'kl_divergence', 'ks_test',
                   'mutual_information', 'sliced_wasserstein_distance']

base_path = "../../Downstream_Results"
data_quality_base_path = "../../Datasets"


def extract_pg_value(file_path, algorithm, missing_rate, metric_name):
    """提取PG值（不区分大小写）"""
    try:
        df = pd.read_csv(file_path)
        target_pattern = f"dirty-{algorithm.lower()}-{missing_rate}.csv"
        matching_rows = df[df['File Name'].str.lower() == target_pattern]
        if not matching_rows.empty:
            return matching_rows[metric_name].values[0]
        return None
    except Exception as e:
        return None


def extract_quality_value(file_path, metric, missing_rate, algorithm):
    """提取数据质量指标值"""
    try:
        df = pd.read_csv(file_path)
        target_file = f"dirty-{algorithm.lower()}-{missing_rate}.csv"
        matching_rows = df[df['file'] == target_file]

        if not matching_rows.empty:
            if metric == "2_wasserstein_distance":
                return matching_rows['2-Wasserstein Distance'].values[0]
            elif metric == "kl_divergence":
                return matching_rows['KL_Divergence'].values[0]
            elif metric == "ks_test":
                return matching_rows['P-Value'].values[0]
            elif metric == "mutual_information":
                return matching_rows['Mutual_Information_target'].values[0]
            elif metric == "sliced_wasserstein_distance":
                return matching_rows['avg_ratio'].values[0]
        return None
    except Exception as e:
        return None


def collect_all_data(task_name, task_config):
    """收集所有数据用于分析"""
    all_data = []
    model = task_config['models'][0]
    metric_name = task_metrics[task_name]['metric']

    print(f"\n收集 {task_name} 任务数据...")

    for dataset in task_config['datasets']:
        print(f"  处理数据集: {dataset}")

        for mechanism in Mechanism:
            perf_file_path = os.path.join(base_path, task_name, dataset, mechanism,
                                          f"{model}-imputation-results-{dataset}.csv")

            if not os.path.exists(perf_file_path):
                print(f"    警告: 性能文件不存在 - {perf_file_path}")
                continue

            for algorithm in Imputation_Algorithms:
                for missing_rate in Missing_rate:
                    # 提取性能指标
                    pg_value = extract_pg_value(perf_file_path, algorithm, missing_rate, metric_name)
                    if pg_value is None:
                        continue

                    # 提取所有数据质量指标
                    quality_dict = {}
                    for q_metric in quality_metrics:
                        quality_file_path = os.path.join(data_quality_base_path, dataset,
                                                         "Data_Quality", q_metric,
                                                         f"{q_metric}_results_{mechanism}.csv")

                        if os.path.exists(quality_file_path):
                            q_value = extract_quality_value(quality_file_path, q_metric,
                                                            missing_rate, algorithm)
                            if q_value is not None:
                                quality_dict[q_metric] = q_value

                    # 如果至少有一个质量指标，则记录数据
                    if quality_dict:
                        record = {
                            'task': task_name,
                            'dataset': dataset,
                            'mechanism': mechanism,
                            'algorithm': algorithm,
                            'missing_rate': missing_rate,
                            'performance': pg_value,
                            **quality_dict
                        }
                        all_data.append(record)

    df = pd.DataFrame(all_data)
    print(f"  收集到 {len(df)} 条有效记录")
    return df


def calculate_correlations_by_group(df, group_by=None):
    """计算分组相关性"""
    results = {}

    if group_by is None:
        groups = {'all': df}
    else:
        groups = {name: group for name, group in df.groupby(group_by)}

    for group_name, group_df in groups.items():
        if len(group_df) < 5:
            continue

        correlations = {}
        for metric in quality_metrics:
            if metric in group_df.columns:
                valid_data = group_df[[metric, 'performance']].dropna()
                if len(valid_data) > 5:
                    pearson_corr, pearson_p = pearsonr(valid_data[metric], valid_data['performance'])
                    spearman_corr, spearman_p = spearmanr(valid_data[metric], valid_data['performance'])

                    correlations[metric] = {
                        'pearson': pearson_corr,
                        'pearson_p': pearson_p,
                        'spearman': spearman_corr,
                        'spearman_p': spearman_p,
                        'n_samples': len(valid_data)
                    }

        results[group_name] = correlations

    return results


def calculate_correlations_by_missing_rate(df):
    """按缺失率计算相关性（缺失率从0.05到0.95，步长0.05）"""
    # 将缺失率转换为小数形式用于显示
    missing_rate_values = [rate / 100 for rate in Missing_rate]  # [0.05, 0.10, ..., 0.95]

    results = {}

    for missing_rate in Missing_rate:
        # 筛选当前缺失率的数据
        rate_df = df[df['missing_rate'] == missing_rate]

        if len(rate_df) < 5:
            continue

        rate_correlations = {}
        for metric in quality_metrics:
            if metric in rate_df.columns:
                valid_data = rate_df[[metric, 'performance']].dropna()
                if len(valid_data) > 5:
                    pearson_corr, pearson_p = pearsonr(valid_data[metric], valid_data['performance'])
                    spearman_corr, spearman_p = spearmanr(valid_data[metric], valid_data['performance'])

                    rate_correlations[metric] = {
                        'pearson': pearson_corr,
                        'pearson_p': pearson_p,
                        'spearman': spearman_corr,
                        'spearman_p': spearman_p,
                        'n_samples': len(valid_data),
                        'missing_rate': missing_rate / 100  # 保存小数形式
                    }

        results[missing_rate / 100] = rate_correlations  # 使用小数作为键名

    return results


def main():
    all_results = {}
    all_tasks_data = []  # 用于存储所有任务的数据

    for task_name, task_config in tasks.items():
        print(f"\n{'=' * 70}")
        print(f"分析任务: {task_name}")
        print(f"{'=' * 70}")

        # 1. 收集数据
        df = collect_all_data(task_name, task_config)

        if len(df) == 0:
            print("没有有效数据，跳过该任务")
            continue

        # 添加到总数据集中
        all_tasks_data.append(df)

        # 2. 整体相关性分析
        print(f"\n【整体相关性分析】")
        overall_corr = calculate_correlations_by_group(df, group_by=None)

        best_metric = None
        best_correlation = -1

        for metric, corr_info in overall_corr['all'].items():
            print(f"\n{metric}:")
            print(f"  Pearson相关系数: {corr_info['pearson']:.4f} (p={corr_info['pearson_p']:.4e})")
            print(f"  Spearman相关系数: {corr_info['spearman']:.4f} (p={corr_info['spearman_p']:.4e})")
            print(f"  样本数: {corr_info['n_samples']}")

            if abs(corr_info['spearman']) > best_correlation:
                best_correlation = abs(corr_info['spearman'])
                best_metric = metric

        print(f"\n★ 与性能相关性最大的指标: {best_metric} (Spearman相关系数: {best_correlation:.4f})")

        # 3. 按缺失机制分析相关性
        print(f"\n【按缺失机制的相关性分析】")
        corr_by_mechanism = calculate_correlations_by_group(df, group_by='mechanism')

        for mechanism in Mechanism:
            if mechanism in corr_by_mechanism:
                print(f"\n{mechanism}:")
                for metric, corr_info in corr_by_mechanism[mechanism].items():
                    print(f"  {metric}: Spearman={corr_info['spearman']:.4f}")

        # 保存结果
        all_results[task_name] = {
            'dataframe': df,
            'overall_correlations': overall_corr['all'],
            'correlations_by_mechanism': corr_by_mechanism,
            'best_metric': best_metric,
            'best_correlation': best_correlation
        }

    # ========== 新增：跨任务综合分析 ==========
    if all_tasks_data:
        print(f"\n{'=' * 70}")
        print("跨任务综合分析（不区分任务类型）")
        print(f"{'=' * 70}")

        # 合并所有任务的数据
        combined_df = pd.concat(all_tasks_data, ignore_index=True)
        print(f"合并后总样本数: {len(combined_df)}")

        # 计算跨任务的整体相关性
        combined_overall_corr = calculate_correlations_by_group(combined_df, group_by=None)

        # 计算跨任务的按机制相关性
        combined_mechanism_corr = calculate_correlations_by_group(combined_df, group_by='mechanism')

        # 计算跨任务的按数据集相关性（可选）
        combined_dataset_corr = calculate_correlations_by_group(combined_df, group_by='dataset')

        # ========== 新增：按缺失率的相关性分析 ==========
        combined_missing_rate_corr = calculate_correlations_by_missing_rate(combined_df)

        # 找出跨任务的最佳指标
        combined_best_metric = None
        combined_best_correlation = -1

        print(f"\n【跨任务整体相关性分析】")
        for metric, corr_info in combined_overall_corr['all'].items():
            print(f"\n{metric}:")
            print(f"  Pearson相关系数: {corr_info['pearson']:.4f} (p={corr_info['pearson_p']:.4e})")
            print(f"  Spearman相关系数: {corr_info['spearman']:.4f} (p={corr_info['spearman_p']:.4e})")
            print(f"  样本数: {corr_info['n_samples']}")

            if abs(corr_info['spearman']) > combined_best_correlation:
                combined_best_correlation = abs(corr_info['spearman'])
                combined_best_metric = metric

        print(
            f"\n★ 跨任务与性能相关性最大的指标: {combined_best_metric} (Spearman相关系数: {combined_best_correlation:.4f})")

        # 跨任务按缺失机制分析
        print(f"\n【跨任务按缺失机制的相关性分析】")
        for mechanism in Mechanism:
            if mechanism in combined_mechanism_corr:
                print(f"\n{mechanism}:")
                for metric, corr_info in combined_mechanism_corr[mechanism].items():
                    print(f"  {metric}: Spearman={corr_info['spearman']:.4f} (样本数={corr_info['n_samples']})")

        # ========== 新增：跨任务按缺失率分析 ==========
        print(f"\n【跨任务按缺失率的相关性分析】")
        print(f"缺失率从5%到95%，每5%一个间隔")
        for missing_rate in sorted(combined_missing_rate_corr.keys()):
            rate_corr = combined_missing_rate_corr[missing_rate]
            if rate_corr:
                print(f"\n缺失率: {missing_rate * 100:.0f}%")
                for metric, corr_info in rate_corr.items():
                    print(f"  {metric}: Spearman={corr_info['spearman']:.4f} (样本数={corr_info['n_samples']})")

        # 将跨任务结果添加到all_results中
        all_results['cross_task'] = {
            'dataframe': combined_df,
            'overall_correlations': combined_overall_corr['all'],
            'correlations_by_mechanism': combined_mechanism_corr,
            'correlations_by_dataset': combined_dataset_corr,
            'correlations_by_missing_rate': combined_missing_rate_corr,  # 新增
            'best_metric': combined_best_metric,
            'best_correlation': combined_best_correlation
        }

    # 保存详细结果到文件
    output_path = os.path.join("results", "correlation_dependence_analysis.txt")
    save_detailed_results(all_results, output_path)
    print(f"\n{'=' * 70}")
    print("分析完成！详细结果已保存到 'correlation_dependence_analysis.txt'")
    print(f"{'=' * 70}")


def save_detailed_results(all_results, output_file):
    """保存详细结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("数据质量指标与性能相关性及依赖性分析报告\n")
        f.write("=" * 80 + "\n\n")

        # 首先输出跨任务综合分析结果（如果存在）
        if 'cross_task' in all_results:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"跨任务综合分析（不区分任务类型）\n")
            f.write(f"{'=' * 80}\n\n")

            results = all_results['cross_task']

            # 最佳指标
            f.write(f"【最佳相关性指标】\n")
            f.write(f"与性能相关性最大的指标: {results['best_metric']}\n")
            f.write(f"Spearman相关系数: {results['best_correlation']:.6f}\n\n")

            # 整体相关性
            f.write(f"【跨任务整体相关性分析】\n")
            f.write(f"{'指标':<35} {'Pearson':<15} {'p值':<15} {'Spearman':<15} {'p值':<15} {'样本数':<10}\n")
            f.write("-" * 105 + "\n")

            for metric, corr_info in results['overall_correlations'].items():
                f.write(f"{metric:<35} {corr_info['pearson']:<15.6f} {corr_info['pearson_p']:<15.4e} "
                        f"{corr_info['spearman']:<15.6f} {corr_info['spearman_p']:<15.4e} {corr_info['n_samples']:<10}\n")

            # 按机制的相关性
            f.write(f"\n【跨任务按缺失机制的相关性分析】\n")
            for mechanism, mech_corr in results['correlations_by_mechanism'].items():
                f.write(f"\n{mechanism}:\n")
                f.write(f"{'指标':<35} {'Pearson':<20} {'Spearman':<20} {'样本数':<10}\n")
                f.write("-" * 85 + "\n")
                for metric, corr_info in mech_corr.items():
                    f.write(
                        f"{metric:<35} {corr_info['pearson']:<20.6f} {corr_info['spearman']:<20.6f} {corr_info['n_samples']:<10}\n")

            # 按数据集的相关性（可选）
            if results.get('correlations_by_dataset'):
                f.write(f"\n【跨任务按数据集的相关性分析】\n")
                for dataset, dataset_corr in results['correlations_by_dataset'].items():
                    if dataset != 'all':
                        f.write(f"\n{dataset}:\n")
                        f.write(f"{'指标':<35} {'Spearman相关系数':<20}\n")
                        f.write("-" * 55 + "\n")
                        for metric, corr_info in dataset_corr.items():
                            f.write(f"{metric:<35} {corr_info['spearman']:<20.6f}\n")

            # ========== 新增：按缺失率的相关性分析 ==========
            if results.get('correlations_by_missing_rate'):
                f.write(f"\n【跨任务按缺失率的相关性分析】\n")
                f.write(f"缺失率从5%到95%，每5%一个间隔\n\n")

                # 创建表格形式展示
                missing_rates = sorted(results['correlations_by_missing_rate'].keys())

                # 表头
                f.write(f"{'缺失率':<10}")
                for metric in quality_metrics:
                    f.write(f"{metric:<30}")
                f.write("\n")
                f.write("-" * (10 + 30 * len(quality_metrics)) + "\n")

                # 为每个缺失率输出一行（Spearman相关系数）
                for missing_rate in missing_rates:
                    rate_corr = results['correlations_by_missing_rate'][missing_rate]
                    f.write(f"{missing_rate * 100:>6.0f}%    ")
                    for metric in quality_metrics:
                        if metric in rate_corr:
                            spearman_val = rate_corr[metric]['spearman']
                            f.write(f"{spearman_val:<30.6f}")
                        else:
                            f.write(f"{'N/A':<30}")
                    f.write("\n")

                # 添加样本数信息
                f.write(f"\n【各缺失率下的样本数】\n")
                f.write(f"{'缺失率':<10}")
                for metric in quality_metrics:
                    f.write(f"{metric:<30}")
                f.write("\n")
                f.write("-" * (10 + 30 * len(quality_metrics)) + "\n")

                for missing_rate in missing_rates:
                    rate_corr = results['correlations_by_missing_rate'][missing_rate]
                    f.write(f"{missing_rate * 100:>6.0f}%    ")
                    for metric in quality_metrics:
                        if metric in rate_corr:
                            n_samples = rate_corr[metric]['n_samples']
                            f.write(f"{n_samples:<30}")
                        else:
                            f.write(f"{'N/A':<30}")
                    f.write("\n")

                # 详细输出每个缺失率下的完整统计信息
                f.write(f"\n【各缺失率下的详细统计信息】\n")
                for missing_rate in missing_rates:
                    rate_corr = results['correlations_by_missing_rate'][missing_rate]
                    if rate_corr:
                        f.write(f"\n缺失率: {missing_rate * 100:.0f}%\n")
                        f.write(
                            f"{'指标':<35} {'Pearson':<15} {'p值':<15} {'Spearman':<15} {'p值':<15} {'样本数':<10}\n")
                        f.write("-" * 105 + "\n")
                        for metric, corr_info in rate_corr.items():
                            f.write(f"{metric:<35} {corr_info['pearson']:<15.6f} {corr_info['pearson_p']:<15.4e} "
                                    f"{corr_info['spearman']:<15.6f} {corr_info['spearman_p']:<15.4e} {corr_info['n_samples']:<10}\n")

            f.write(f"\n{'=' * 80}\n\n")

        # 然后输出各任务单独的分析结果
        for task_name, results in all_results.items():
            if task_name == 'cross_task':
                continue

            f.write(f"\n{'=' * 80}\n")
            f.write(f"任务类型: {task_name}\n")
            f.write(f"{'=' * 80}\n\n")

            # 最佳指标
            f.write(f"【最佳相关性指标】\n")
            f.write(f"与性能相关性最大的指标: {results['best_metric']}\n")
            f.write(f"Spearman相关系数: {results['best_correlation']:.6f}\n\n")

            # 整体相关性
            f.write(f"【整体相关性分析】\n")
            f.write(f"{'指标':<35} {'Pearson':<15} {'p值':<15} {'Spearman':<15} {'p值':<15} {'样本数':<10}\n")
            f.write("-" * 105 + "\n")

            for metric, corr_info in results['overall_correlations'].items():
                f.write(f"{metric:<35} {corr_info['pearson']:<15.6f} {corr_info['pearson_p']:<15.4e} "
                        f"{corr_info['spearman']:<15.6f} {corr_info['spearman_p']:<15.4e} {corr_info['n_samples']:<10}\n")

            # 按机制的相关性
            f.write(f"\n【按缺失机制的相关性分析】\n")
            for mechanism, mech_corr in results['correlations_by_mechanism'].items():
                f.write(f"\n{mechanism}:\n")
                f.write(f"{'指标':<35} {'Pearson':<20} {'Spearman':<20} {'样本数':<10}\n")
                f.write("-" * 85 + "\n")
                for metric, corr_info in mech_corr.items():
                    f.write(
                        f"{metric:<35} {corr_info['pearson']:<20.6f} {corr_info['spearman']:<20.6f} {corr_info['n_samples']:<10}\n")

        f.write(f"\n{'=' * 80}\n")
        f.write("分析报告结束\n")
        f.write("=" * 80 + "\n")


if __name__ == "__main__":
    main()