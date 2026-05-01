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
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
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
    处理单个任务
    """
    results = {}
    datasets = task_config['datasets']
    model = task_config['models'][0]
    metric_name = task_metrics[task_name]['metric']

    for dataset in datasets:
        print(f"Processing {task_name} - {dataset}...")
        results[dataset] = {}

        for algorithm in Imputation_Algorithms:
            results[dataset][algorithm] = {}

            # 存储不同机制下的PG值
            mechanism_values = {mech: [] for mech in Mechanism}

            # 读取每个缺失率下的数据
            for missing_rate in Missing_rate:
                for mechanism in Mechanism:
                    # 构建文件路径
                    file_path = os.path.join(base_path, task_name, dataset, mechanism,
                                             f"{model}-imputation-results-{dataset}.csv")

                    if os.path.exists(file_path):
                        pg_value = extract_pg_value(file_path, algorithm, missing_rate, metric_name)
                        if pg_value is not None:
                            mechanism_values[mechanism].append(pg_value)
                    else:
                        print(f"  Warning: File not found - {file_path}")

            # 计算平均差值（MAR - MCAR 和 MNAR - MCAR）
            if len(mechanism_values['MCAR']) == len(Missing_rate):
                mcar_values = np.array(mechanism_values['MCAR'])

                # 计算MAR和MNAR相对于MCAR的平均差值
                for mech in ['MAR', 'MNAR']:
                    if len(mechanism_values[mech]) == len(Missing_rate):
                        mech_values = np.array(mechanism_values[mech])
                        # 计算逐点差值，然后取平均
                        diff_values = mech_values - mcar_values
                        mean_diff = np.mean(diff_values)
                        results[dataset][algorithm][mech] = mean_diff
                    else:
                        results[dataset][algorithm][mech] = None
                        print(f"  Warning: {mech} data missing for {algorithm}")
            else:
                results[dataset][algorithm]['MAR'] = None
                results[dataset][algorithm]['MNAR'] = None
                print(f"  Warning: MCAR data missing for {algorithm}")

    return results


def compute_averages(results, task_name):
    """
    计算平均值（基于平均差值）
    """
    # 数据集间平均
    dataset_avg_mar = {}
    dataset_avg_mnar = {}

    for dataset in results:
        dataset_mar = [results[dataset][alg]['MAR'] for alg in results[dataset]
                       if results[dataset][alg]['MAR'] is not None]
        dataset_mnar = [results[dataset][alg]['MNAR'] for alg in results[dataset]
                        if results[dataset][alg]['MNAR'] is not None]
        dataset_avg_mar[dataset] = np.mean(dataset_mar) if dataset_mar else 0
        dataset_avg_mnar[dataset] = np.mean(dataset_mnar) if dataset_mnar else 0

    # 填补算法间平均
    algorithm_avg_mar = {}
    algorithm_avg_mnar = {}

    for dataset in results:
        for algorithm in results[dataset]:
            if algorithm not in algorithm_avg_mar:
                algorithm_avg_mar[algorithm] = []
                algorithm_avg_mnar[algorithm] = []
            if results[dataset][algorithm]['MAR'] is not None:
                algorithm_avg_mar[algorithm].append(results[dataset][algorithm]['MAR'])
            if results[dataset][algorithm]['MNAR'] is not None:
                algorithm_avg_mnar[algorithm].append(results[dataset][algorithm]['MNAR'])

    # 计算每个算法的平均
    for algorithm in algorithm_avg_mar:
        algorithm_avg_mar[algorithm] = np.mean(algorithm_avg_mar[algorithm]) if algorithm_avg_mar[algorithm] else 0
        algorithm_avg_mnar[algorithm] = np.mean(algorithm_avg_mnar[algorithm]) if algorithm_avg_mnar[algorithm] else 0

    # 总平均
    all_mar_diffs = []
    all_mnar_diffs = []

    for dataset in results:
        for algorithm in results[dataset]:
            if results[dataset][algorithm]['MAR'] is not None:
                all_mar_diffs.append(results[dataset][algorithm]['MAR'])
            if results[dataset][algorithm]['MNAR'] is not None:
                all_mnar_diffs.append(results[dataset][algorithm]['MNAR'])

    total_avg_mar = np.mean(all_mar_diffs) if all_mar_diffs else 0
    total_avg_mnar = np.mean(all_mnar_diffs) if all_mnar_diffs else 0

    return {
        'dataset_avg_mar': dataset_avg_mar,
        'dataset_avg_mnar': dataset_avg_mnar,
        'algorithm_avg_mar': algorithm_avg_mar,
        'algorithm_avg_mnar': algorithm_avg_mnar,
        'total_avg_mar': total_avg_mar,
        'total_avg_mnar': total_avg_mnar
    }


def save_results_to_file(all_results, output_file="analysis_results.txt"):
    """
    将结果保存到txt文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("缺失机制比较分析结果（平均差值：MAR-MCAR 和 MNAR-MCAR）\n")
        f.write("=" * 80 + "\n\n")

        for task_name, results in all_results.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"任务: {task_name}\n")
            f.write(f"{'=' * 80}\n")

            # 计算平均值
            averages = compute_averages(results, task_name)

            # 写入数据集平均结果
            f.write(f"\n--- 数据集平均结果 ---\n")
            f.write(f"{'数据集':<20} {'MAR平均差值':<20} {'MNAR平均差值':<20}\n")
            f.write("-" * 60 + "\n")
            for dataset in averages['dataset_avg_mar']:
                f.write(
                    f"{dataset:<20} {averages['dataset_avg_mar'][dataset]:<20.6f} {averages['dataset_avg_mnar'][dataset]:<20.6f}\n")

            # 写入填补算法平均结果
            f.write(f"\n--- 填补算法平均结果 ---\n")
            f.write(f"{'算法':<15} {'MAR平均差值':<20} {'MNAR平均差值':<20}\n")
            f.write("-" * 55 + "\n")
            for algorithm in averages['algorithm_avg_mar']:
                f.write(
                    f"{algorithm:<15} {averages['algorithm_avg_mar'][algorithm]:<20.6f} {averages['algorithm_avg_mnar'][algorithm]:<20.6f}\n")

            # 写入总平均
            f.write(f"\n--- 总平均结果 ---\n")
            f.write(f"MAR总平均差值: {averages['total_avg_mar']:.6f}\n")
            f.write(f"MNAR总平均差值: {averages['total_avg_mnar']:.6f}\n")

            # 写入详细结果
            f.write(f"\n--- 详细结果 ---\n")
            for dataset in results:
                f.write(f"\n数据集: {dataset}\n")
                f.write(f"{'算法':<15} {'MAR差值':<20} {'MNAR差值':<20}\n")
                f.write("-" * 55 + "\n")
                for algorithm in results[dataset]:
                    mar_val = results[dataset][algorithm]['MAR'] if results[dataset][algorithm][
                                                                        'MAR'] is not None else 'N/A'
                    mnar_val = results[dataset][algorithm]['MNAR'] if results[dataset][algorithm][
                                                                          'MNAR'] is not None else 'N/A'
                    f.write(f"{algorithm:<15} {str(mar_val):<20} {str(mnar_val):<20}\n")

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
    output_file = os.path.join("results", "missing_mechanism_analysis_results2.txt")
    save_results_to_file(all_results, output_file)
    print("\n分析完成！结果已保存到 'missing_mechanism_analysis_results2.txt'")


if __name__ == "__main__":
    main()