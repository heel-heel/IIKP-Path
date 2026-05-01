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
    处理单个任务，返回平均值和中位数两种结果
    """
    results_mean = {}
    results_median = {}
    datasets = task_config['datasets']
    model = task_config['models'][0]
    metric_name = task_metrics[task_name]['metric']

    for dataset in datasets:
        print(f"Processing {task_name} - {dataset}...")
        results_mean[dataset] = {}
        results_median[dataset] = {}

        for algorithm in Imputation_Algorithms:
            results_mean[dataset][algorithm] = {}
            results_median[dataset][algorithm] = {}

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

            # 计算差值的平均值和中位数
            if len(mechanism_values['MCAR']) == len(Missing_rate):
                mcar_values = np.array(mechanism_values['MCAR'])
                mcar_mean = np.mean(mcar_values)
                mcar_median = np.median(mcar_values)

                # 计算MAR和MNAR相对于MCAR的差值
                for mech in ['MAR', 'MNAR']:
                    if len(mechanism_values[mech]) == len(Missing_rate):
                        mech_values = np.array(mechanism_values[mech])
                        diff_values = mech_values - mcar_values

                        # 平均值版本
                        diff_sum = np.sum(diff_values)
                        ratio_mean = diff_sum / mcar_mean if mcar_mean != 0 else 0
                        results_mean[dataset][algorithm][mech] = ratio_mean

                        # 中位数版本（使用差值的中位数 / mcar中位数）
                        diff_median = np.median(diff_values)
                        ratio_median = diff_median / mcar_median if mcar_median != 0 else 0
                        results_median[dataset][algorithm][mech] = ratio_median
                    else:
                        results_mean[dataset][algorithm][mech] = None
                        results_median[dataset][algorithm][mech] = None
                        print(f"  Warning: {mech} data missing for {algorithm}")
            else:
                results_mean[dataset][algorithm]['MAR'] = None
                results_mean[dataset][algorithm]['MNAR'] = None
                results_median[dataset][algorithm]['MAR'] = None
                results_median[dataset][algorithm]['MNAR'] = None
                print(f"  Warning: MCAR data missing for {algorithm}")

    return results_mean, results_median


def compute_averages(results, task_name):
    """
    计算平均值（基于平均差值比例）
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
    all_mar_ratios = []
    all_mnar_ratios = []

    for dataset in results:
        for algorithm in results[dataset]:
            if results[dataset][algorithm]['MAR'] is not None:
                all_mar_ratios.append(results[dataset][algorithm]['MAR'])
            if results[dataset][algorithm]['MNAR'] is not None:
                all_mnar_ratios.append(results[dataset][algorithm]['MNAR'])

    total_avg_mar = np.mean(all_mar_ratios) if all_mar_ratios else 0
    total_avg_mnar = np.mean(all_mnar_ratios) if all_mnar_ratios else 0

    return {
        'dataset_avg_mar': dataset_avg_mar,
        'dataset_avg_mnar': dataset_avg_mnar,
        'algorithm_avg_mar': algorithm_avg_mar,
        'algorithm_avg_mnar': algorithm_avg_mnar,
        'total_avg_mar': total_avg_mar,
        'total_avg_mnar': total_avg_mnar
    }


def compute_medians(results, task_name):
    """
    计算中位数（基于中位数差值比例）
    """
    # 数据集间中位数
    dataset_median_mar = {}
    dataset_median_mnar = {}

    for dataset in results:
        dataset_mar = [results[dataset][alg]['MAR'] for alg in results[dataset]
                       if results[dataset][alg]['MAR'] is not None]
        dataset_mnar = [results[dataset][alg]['MNAR'] for alg in results[dataset]
                        if results[dataset][alg]['MNAR'] is not None]
        dataset_median_mar[dataset] = np.median(dataset_mar) if dataset_mar else 0
        dataset_median_mnar[dataset] = np.median(dataset_mnar) if dataset_mnar else 0

    # 填补算法间中位数
    algorithm_median_mar = {}
    algorithm_median_mnar = {}

    for dataset in results:
        for algorithm in results[dataset]:
            if algorithm not in algorithm_median_mar:
                algorithm_median_mar[algorithm] = []
                algorithm_median_mnar[algorithm] = []
            if results[dataset][algorithm]['MAR'] is not None:
                algorithm_median_mar[algorithm].append(results[dataset][algorithm]['MAR'])
            if results[dataset][algorithm]['MNAR'] is not None:
                algorithm_median_mnar[algorithm].append(results[dataset][algorithm]['MNAR'])

    # 计算每个算法的中位数
    for algorithm in algorithm_median_mar:
        algorithm_median_mar[algorithm] = np.median(algorithm_median_mar[algorithm]) if algorithm_median_mar[
            algorithm] else 0
        algorithm_median_mnar[algorithm] = np.median(algorithm_median_mnar[algorithm]) if algorithm_median_mnar[
            algorithm] else 0

    # 总中位数
    all_mar_ratios = []
    all_mnar_ratios = []

    for dataset in results:
        for algorithm in results[dataset]:
            if results[dataset][algorithm]['MAR'] is not None:
                all_mar_ratios.append(results[dataset][algorithm]['MAR'])
            if results[dataset][algorithm]['MNAR'] is not None:
                all_mnar_ratios.append(results[dataset][algorithm]['MNAR'])

    total_median_mar = np.median(all_mar_ratios) if all_mar_ratios else 0
    total_median_mnar = np.median(all_mnar_ratios) if all_mnar_ratios else 0

    return {
        'dataset_median_mar': dataset_median_mar,
        'dataset_median_mnar': dataset_median_mnar,
        'algorithm_median_mar': algorithm_median_mar,
        'algorithm_median_mnar': algorithm_median_mnar,
        'total_median_mar': total_median_mar,
        'total_median_mnar': total_median_mnar
    }


def save_results_to_file(all_results_mean, all_results_median, output_file="analysis_results.txt"):
    """
    将结果保存到txt文件，同时输出平均值和中位数
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("缺失机制比较分析结果\n")
        f.write("=" * 80 + "\n\n")

        for task_name in all_results_mean.keys():
            results_mean = all_results_mean[task_name]
            results_median = all_results_median[task_name]

            f.write(f"\n{'=' * 80}\n")
            f.write(f"任务: {task_name}\n")
            f.write(f"{'=' * 80}\n")

            # 计算平均值和中位数统计
            averages = compute_averages(results_mean, task_name)
            medians = compute_medians(results_median, task_name)

            # ========== 平均值部分 ==========
            f.write(f"\n{'=' * 40}\n")
            f.write("【平均值统计】\n")
            f.write(f"{'=' * 40}\n")

            # 写入数据集平均结果
            f.write(f"\n--- 数据集平均结果 ---\n")
            f.write(f"{'数据集':<20} {'MAR平均比值':<20} {'MNAR平均比值':<20}\n")
            f.write("-" * 60 + "\n")
            for dataset in averages['dataset_avg_mar']:
                f.write(
                    f"{dataset:<20} {averages['dataset_avg_mar'][dataset]:<20.6f} {averages['dataset_avg_mnar'][dataset]:<20.6f}\n")

            # 写入填补算法平均结果
            f.write(f"\n--- 填补算法平均结果 ---\n")
            f.write(f"{'算法':<15} {'MAR平均比值':<20} {'MNAR平均比值':<20}\n")
            f.write("-" * 55 + "\n")
            for algorithm in averages['algorithm_avg_mar']:
                f.write(
                    f"{algorithm:<15} {averages['algorithm_avg_mar'][algorithm]:<20.6f} {averages['algorithm_avg_mnar'][algorithm]:<20.6f}\n")

            # 写入总平均
            f.write(f"\n--- 总平均结果 ---\n")
            f.write(f"MAR总平均比值: {averages['total_avg_mar']:.6f}\n")
            f.write(f"MNAR总平均比值: {averages['total_avg_mnar']:.6f}\n")

            # ========== 中位数部分 ==========
            f.write(f"\n{'=' * 40}\n")
            f.write("【中位数统计】\n")
            f.write(f"{'=' * 40}\n")

            # 写入数据集中位数结果
            f.write(f"\n--- 数据集中位数结果 ---\n")
            f.write(f"{'数据集':<20} {'MAR中位数比值':<20} {'MNAR中位数比值':<20}\n")
            f.write("-" * 60 + "\n")
            for dataset in medians['dataset_median_mar']:
                f.write(
                    f"{dataset:<20} {medians['dataset_median_mar'][dataset]:<20.6f} {medians['dataset_median_mnar'][dataset]:<20.6f}\n")

            # 写入填补算法中位数结果
            f.write(f"\n--- 填补算法中位数结果 ---\n")
            f.write(f"{'算法':<15} {'MAR中位数比值':<20} {'MNAR中位数比值':<20}\n")
            f.write("-" * 55 + "\n")
            for algorithm in medians['algorithm_median_mar']:
                f.write(
                    f"{algorithm:<15} {medians['algorithm_median_mar'][algorithm]:<20.6f} {medians['algorithm_median_mnar'][algorithm]:<20.6f}\n")

            # 写入总中位数
            f.write(f"\n--- 总中位数结果 ---\n")
            f.write(f"MAR总中位数比值: {medians['total_median_mar']:.6f}\n")
            f.write(f"MNAR总中位数比值: {medians['total_median_mnar']:.6f}\n")

            # ========== 详细结果（平均值） ==========
            f.write(f"\n{'=' * 40}\n")
            f.write("【详细结果 - 平均值】\n")
            f.write(f"{'=' * 40}\n")
            for dataset in results_mean:
                f.write(f"\n数据集: {dataset}\n")
                f.write(f"{'算法':<15} {'MAR比值(平均)':<20} {'MNAR比值(平均)':<20}\n")
                f.write("-" * 55 + "\n")
                for algorithm in results_mean[dataset]:
                    mar_val = results_mean[dataset][algorithm]['MAR'] if results_mean[dataset][algorithm][
                                                                             'MAR'] is not None else 'N/A'
                    mnar_val = results_mean[dataset][algorithm]['MNAR'] if results_mean[dataset][algorithm][
                                                                               'MNAR'] is not None else 'N/A'
                    f.write(f"{algorithm:<15} {str(mar_val):<20} {str(mnar_val):<20}\n")

            # ========== 详细结果（中位数） ==========
            f.write(f"\n{'=' * 40}\n")
            f.write("【详细结果 - 中位数】\n")
            f.write(f"{'=' * 40}\n")
            for dataset in results_median:
                f.write(f"\n数据集: {dataset}\n")
                f.write(f"{'算法':<15} {'MAR比值(中位数)':<20} {'MNAR比值(中位数)':<20}\n")
                f.write("-" * 55 + "\n")
                for algorithm in results_median[dataset]:
                    mar_val = results_median[dataset][algorithm]['MAR'] if results_median[dataset][algorithm][
                                                                               'MAR'] is not None else 'N/A'
                    mnar_val = results_median[dataset][algorithm]['MNAR'] if results_median[dataset][algorithm][
                                                                                 'MNAR'] is not None else 'N/A'
                    f.write(f"{algorithm:<15} {str(mar_val):<20} {str(mnar_val):<20}\n")

        f.write(f"\n{'=' * 80}\n")
        f.write("分析完成\n")
        f.write("=" * 80 + "\n")


# 主程序
def main():
    all_results_mean = {}
    all_results_median = {}

    for task_name, task_config in tasks.items():
        print(f"\n处理任务: {task_name}")
        results_mean, results_median = process_task(task_name, task_config)
        all_results_mean[task_name] = results_mean
        all_results_median[task_name] = results_median

    # 创建results目录（如果不存在）
    os.makedirs("results", exist_ok=True)

    # 保存结果到文件
    output_file = os.path.join("results", "missing_mechanism_analysis_results.txt")
    save_results_to_file(all_results_mean, all_results_median, output_file)
    print("\n分析完成！结果已保存到 'missing_mechanism_analysis_results.txt'")


if __name__ == "__main__":
    main()