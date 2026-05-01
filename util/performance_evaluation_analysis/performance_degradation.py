# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

# 配置参数
tasks = {
    'timeseries': {
        'datasets': ['ETTh1', 'ETTm1', 'Illness', 'Exchange', 'Weather'],
        'models': ['mlp'],
    },
    'classification': {
        'datasets': ['Beers', 'Flights', 'Hospital', 'RedWineQuality', 'AvocadoRipeness'],
        'models': ['mlp']
    },
    'regression': {
        'datasets': ['concrete', 'CCPP', 'AirfoilSelfNoise', 'Abalone', 'ParisHousing'],
        'models': ['mlp']
    }
}

Imputation_Algorithms = ['Mean', 'Median', 'Mode', 'KNN', 'HDI', 'MICE', 'IIM', 'SI',
                         'MFI', 'MissFI', 'XGBI', 'GAIN', 'MIDAE']
# 只使用缺失率95%
TARGET_MISSING_RATE = 95
task_metrics = {
    'classification': {'metric': 'PG(F1 Score)'},
    'timeseries': {'metric': 'PG(RMSE)'},
    'regression': {'metric': 'PG(MAE)'}
}
Mechanism = ['MCAR', 'MAR', 'MNAR']

base_path = "../../Downstream_Results"


def extract_pg_value(file_path, algorithm, missing_rate, metric_name):
    """
    从CSV文件中提取指定填补算法和缺失率的PG值（不区分大小写）
    """
    try:
        df = pd.read_csv(file_path)
        # 查找对应的行 - 不区分大小写匹配
        target_pattern = f"dirty-{algorithm.lower()}-{missing_rate}.csv"

        # 将File Name列转换为小写进行比较
        matching_rows = df[df['File Name'].str.lower() == target_pattern]

        if not matching_rows.empty:
            return matching_rows[metric_name].values[0]
        else:
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def process_task(task_name, task_config):
    """
    处理单个任务，只提取缺失率为95%时的PG值（跨所有缺失机制）
    """
    results = {}
    datasets = task_config['datasets']
    model = task_config['models'][0]
    metric_name = task_metrics[task_name]['metric']

    for dataset in datasets:
        print(f"Processing {task_name} - {dataset}...")
        results[dataset] = {}

        for algorithm in Imputation_Algorithms:
            # 存储所有机制下的PG值（合并在一起）
            all_pg_values = []

            # 遍历三种缺失机制
            for mechanism in Mechanism:
                # 构建文件路径
                file_path = os.path.join(base_path, task_name, dataset, mechanism,
                                         f"{model}-imputation-results-{dataset}.csv")

                if os.path.exists(file_path):
                    pg_value = extract_pg_value(file_path, algorithm, TARGET_MISSING_RATE, metric_name)
                    if pg_value is not None:
                        all_pg_values.append(pg_value)
                    else:
                        print(f"  Warning: {algorithm} at {TARGET_MISSING_RATE}% missing in {mechanism} for {dataset}")
                else:
                    print(f"  Warning: File not found - {file_path}")

            # 存储该算法在该数据集上的所有PG值（跨三种机制）
            results[dataset][algorithm] = all_pg_values

    return results


def compute_statistics(results, task_name):
    """
    计算PG的平均值（跨填补算法、数据集和缺失机制）
    """
    # 1. 每个数据集上，跨所有填补算法和所有缺失机制的平均PG
    dataset_avg = {}
    for dataset in results:
        all_values = []
        for algorithm in results[dataset]:
            all_values.extend(results[dataset][algorithm])  # 合并该数据集下所有算法的PG值
        dataset_avg[dataset] = np.mean(all_values) if all_values else 0

    # 2. 每个填补算法上，跨所有数据集和所有缺失机制的平均PG
    algorithm_avg = {}
    for dataset in results:
        for algorithm in results[dataset]:
            if algorithm not in algorithm_avg:
                algorithm_avg[algorithm] = []
            algorithm_avg[algorithm].extend(results[dataset][algorithm])

    # 计算每个算法的平均值
    for algorithm in algorithm_avg:
        algorithm_avg[algorithm] = np.mean(algorithm_avg[algorithm]) if algorithm_avg[algorithm] else 0

    # 3. 总平均（跨所有数据集、所有填补算法和所有缺失机制）
    all_values_total = []
    for dataset in results:
        for algorithm in results[dataset]:
            all_values_total.extend(results[dataset][algorithm])
    total_avg = np.mean(all_values_total) if all_values_total else 0

    # 4. 每个数据集上，每个填补算法的平均PG（跨三种机制）
    detail_avg = {}
    for dataset in results:
        detail_avg[dataset] = {}
        for algorithm in results[dataset]:
            detail_avg[dataset][algorithm] = np.mean(results[dataset][algorithm]) if results[dataset][algorithm] else 0

    return {
        'dataset_avg': dataset_avg,
        'algorithm_avg': algorithm_avg,
        'total_avg': total_avg,
        'detail_avg': detail_avg
    }


def save_results_to_file(all_results, output_file="analysis_results.txt"):
    """
    将结果保存到txt文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"缺失率为 {TARGET_MISSING_RATE}% 时 PG 平均值分析结果（跨三种缺失机制）\n")
        f.write("=" * 80 + "\n\n")

        for task_name, results in all_results.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"任务: {task_name}\n")
            f.write(f"{'=' * 80}\n")

            # 计算统计量
            stats = compute_statistics(results, task_name)

            # ========== 每个数据集上跨所有填补算法和缺失机制的平均PG ==========
            f.write(f"\n{'=' * 60}\n")
            f.write("【每个数据集上跨所有填补算法和缺失机制的平均PG】\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"{'数据集':<20} {'平均PG':<20}\n")
            f.write("-" * 40 + "\n")
            for dataset in stats['dataset_avg']:
                f.write(f"{dataset:<20} {stats['dataset_avg'][dataset]:<20.6f}\n")

            # ========== 每个填补算法上跨所有数据集和缺失机制的平均PG ==========
            f.write(f"\n{'=' * 60}\n")
            f.write("【每个填补算法上跨所有数据集和缺失机制的平均PG】\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"{'算法':<15} {'平均PG':<20}\n")
            f.write("-" * 35 + "\n")
            # 按平均PG值排序
            sorted_algorithms = sorted(stats['algorithm_avg'].items(), key=lambda x: x[1], reverse=True)
            for algorithm, avg_value in sorted_algorithms:
                f.write(f"{algorithm:<15} {avg_value:<20.6f}\n")

            # ========== 总平均 ==========
            f.write(f"\n{'=' * 60}\n")
            f.write("【总平均PG（跨所有数据集、所有填补算法和所有缺失机制）】\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"总平均PG: {stats['total_avg']:.6f}\n")

            # ========== 详细结果（每个数据集、每个算法的平均PG） ==========
            f.write(f"\n{'=' * 60}\n")
            f.write("【详细结果（每个数据集、每个算法，跨三种缺失机制的平均PG）】\n")
            f.write(f"{'=' * 60}\n")

            for dataset in results:
                f.write(f"\n数据集: {dataset}\n")
                f.write(f"{'算法':<15} {'平均PG（跨MCAR/MAR/MNAR）':<30}\n")
                f.write("-" * 45 + "\n")
                # 按平均PG值排序
                sorted_algorithms_detail = sorted(stats['detail_avg'][dataset].items(), key=lambda x: x[1], reverse=True)
                for algorithm, avg_value in sorted_algorithms_detail:
                    f.write(f"{algorithm:<15} {avg_value:<30.6f}\n")

        f.write(f"\n{'=' * 80}\n")
        f.write("分析完成\n")
        f.write("=" * 80 + "\n")


# 主程序
def main():
    all_results = {}

    for task_name, task_config in tasks.items():
        print(f"\n处理任务: {task_name}")
        results = process_task(task_name, task_config)
        all_results[task_name] = results

    # 创建results目录（如果不存在）
    os.makedirs("results", exist_ok=True)

    # 保存结果到文件
    output_file = os.path.join("results", f"pg_degradation_analysis.txt")
    save_results_to_file(all_results, output_file)
    print(f"\n分析完成！结果已保存到 '{output_file}'")


if __name__ == "__main__":
    main()