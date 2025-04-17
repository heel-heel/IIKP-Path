import numpy as np
import pandas as pd
import os
import sys

# 定义数据集的配置信息和填补算法列表
datasets_numerical = {
    "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"},
}
datasets_categorical = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    "Flights": {"target_column": "flight", "unrelated_column": None},
    "Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"}
}
Imputation_Algorithms_Numerical = ['mean', 'median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'rf', 'xgbi', 'gain', 'midae']
#Imputation_Algorithms_Categorical = ['mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'rf', 'xgbi', 'gain', 'midae']
Imputation_Algorithms_Categorical = ['knn', 'hdi', 'mice', 'iim', 'si', 'rf', 'xgbi', 'gain', 'midae']


# 定义计算目标列缺失率的函数
def calculate_missing_rate(input_missing_data, target_column):
    total_cells = input_missing_data[target_column].size  # 目标列的总单元格数
    missing_cells = input_missing_data[target_column].isnull().sum()  # 目标列的缺失单元格数
    missing_rate = missing_cells / total_cells  # 目标列的缺失率
    return missing_rate


# 定义数据质量评估函数
def evaluate_data_quality(missing_rate, model, data_quality_threshold):
    print(f"正在进行{model}的数据质量评估...")
    history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
    base_path = "../Datasets"
    for metric, threshold in data_quality_threshold.items():
        if threshold is None:
            continue
        for dataset in datasets_numerical:
            #data_quality_file = pd.read_csv(f"../Datasets/{dataset}/Data_Quality/{metric}/{metric}_results.csv")
            data_quality_file = pd.read_csv(os.path.join(base_path, dataset, "Data_Quality", metric, f"{metric}_results.csv"))
            row = data_quality_file[data_quality_file['file'] == history_imputed_file]
            if row.empty:
                return False  # 如果没有找到对应的行，返回False
            if metric == "2_wasserstein_distance":
                value = row['2-Wasserstein Distance'].values[0]
                if value > threshold:
                    print(f"{model}因为在数据集{dataset}上的{metric}超过阈值而被排除")
                    return False
            elif metric == "kl_divergence":
                value = row['KL_Divergence'].values[0]
                if value > threshold:
                    print(f"{model}因为在数据集{dataset}上的{metric}超过阈值而被排除")
                    return False
            elif metric == "ks_test":
                value = row['P-Value'].values[0]
                if value < threshold:
                    print(f"{model}因为在数据集{dataset}上的{metric}超过阈值而被排除")
                    return False
            elif metric == "mutual_information":
                value = row['Mutual_Information_target'].values[0]
                if value < threshold:
                    print(f"{model}因为在数据集{dataset}上的{metric}超过阈值而被排除")
                    return False
            elif metric == "sliced_wasserstein_distance":
                value = row['avg_ratio'].values[0]
                if value > threshold:
                    print(f"{model}因为在数据集{dataset}上的{metric}超过阈值而被排除")
                    return False
            else:
                raise ValueError(f"Unsupported metric: {metric}")
    return True


# 定义关键因素评估函数
def evaluate_key_factors(missing_rate, model, input_task_type, key_factors_threshold):
    print(f"正在进行{model}的关键因素评估...")
    history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
    base_path = "../Datasets"
    if input_task_type == "timeseries":  # 对于时序预测任务，评估残差等关键因素
        for metric, threshold in key_factors_threshold.items():
            if threshold is None:
                continue
            print(f"---正在进行{metric}评估...")
            for dataset in datasets_numerical:
                key_factors_file = pd.read_csv(os.path.join(base_path, dataset, "Mechanism", "timeseries", "decompose_basic", "decompose_basic_results.csv"))
                row = key_factors_file[key_factors_file['file'] == history_imputed_file]
                if row.empty:
                    return False  # 如果没有找到对应的行，返回False
                if metric == "trend":
                    value = row['Corr_Trend'].values[0]
                elif metric == "seasonal":
                    value = row['Corr_Seasonal'].values[0]
                elif metric == "resid":
                    value = row['Corr_Residuals'].values[0]
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                if value < threshold:
                    return False
        return True
    elif input_task_type == "classification":  # 对于分类任务，评估标签正确比例和类别可分性等关键因素
        for metric, threshold in key_factors_threshold.items():
            if threshold is None:
                continue
            print(f"---正在进行{metric}评估...")
            for dataset in datasets_categorical:
                key_factors_file = pd.read_csv(os.path.join(base_path, dataset, "Mechanism", "classification", f"{metric}_results.csv"))
                row = key_factors_file[key_factors_file['file'] == history_imputed_file]
                if row.empty:
                    return False  # 如果没有找到对应的行，返回False
                if metric == "label_correctness_ratio":
                    value = row['Consistent Rate'].values[0]
                elif metric == "class_discriminability":
                    value = row['J Value'].values[0]
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                if value < threshold:
                    return False
        return True
    elif input_task_type == "regression":  # 对于回归任务，评估特征-目标相关性等关键因素
        for metric, threshold in key_factors_threshold.items():
            if threshold is None:
                continue
            print(f"---正在进行{metric}评估...")
            for dataset in datasets_numerical:
                key_factors_file = pd.read_csv(os.path.join(base_path, dataset, "Mechanism", "regression", metric, f"{metric}_results.csv"))
                row = key_factors_file[key_factors_file['file'] == history_imputed_file]
                if row.empty:
                    return False  # 如果没有找到对应的行，返回False
                if metric == "imputation_deviation":
                    value = row['deviation_radio'].values[0]
                    if value > threshold:
                        return False
                elif metric == "feature_target_corr":
                    value = row['Avg_Correlation'].values[0]
                    if value < threshold:
                        return False
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
        return True
    else:
        raise ValueError("Unsupported task type")


def evaluate_downstream_task_performance(missing_rate, model, input_task_type):
    print("正在进行下游分析任务性能评估...")
    history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
    base_path = "../Downstream_Results"
    if input_task_type == "timeseries":
        for dataset in datasets_numerical:
            downstream_task_performance_file = pd.read_csv(os.path.join(base_path, input_task_type, dataset, "mlp-imputation-results-{dataset}.csv"))
            row = downstream_task_performance_file[
                downstream_task_performance_file['File Name'] == history_imputed_file]
            value = row['PG'].values[0]
            return value
    elif input_task_type == "classification":
        for dataset in datasets_categorical:
            downstream_task_performance_file = pd.read_csv(os.path.join(base_path, input_task_type, dataset, "mlp-imputation-results-{dataset}.csv"))
            row = downstream_task_performance_file[
                downstream_task_performance_file['File Name'] == history_imputed_file]
            value = row['PG'].values[0]
            return value
    elif input_task_type == "regression":
        for dataset in datasets_numerical:
            downstream_task_performance_file = pd.read_csv(os.path.join(base_path, input_task_type, dataset, "mlp-imputation-results-{dataset}.csv"))
            row = downstream_task_performance_file[
                downstream_task_performance_file['File Name'] == history_imputed_file]
            value = row['PG'].values[0]
            return value


# 定义填补策略选择函数
def select_imputation_strategy(missing_rate, input_task_type, performance_threshold, data_quality_threshold, key_factors_threshold):
    if input_task_type == "timeseries_forcasting" or input_task_type == "regression":
        Alternative_algorithms = Imputation_Algorithms_Numerical
    elif input_task_type == "classification":
        Alternative_algorithms = Imputation_Algorithms_Categorical
    selected_methods_data_quality = []
    selected_methods_key_factors = []
    selected_methods_final = []
    for model in Alternative_algorithms:
        if not evaluate_data_quality(missing_rate, model, data_quality_threshold):
            continue  # 数据质量未达到阈值，剪枝，尝试下一个填补算法
        else:
            print(f"{model}通过数据质量评估")
            selected_methods_data_quality.append(model)
        if not evaluate_key_factors(missing_rate, model, input_task_type, key_factors_threshold):
            continue  # 关键因素未达到阈值，剪枝，尝试下一个填补算法
        else:
            selected_methods_key_factors.append(model)
        #performance_metric = evaluate_downstream_task_performance(missing_rate, model, input_task_type)
        #if performance_metric <= performance_threshold:
        selected_methods_final.append(model)  # 性能达到阈值，将当前填补算法加入列表
    if selected_methods_data_quality:
        print(f"Selected imputation methods after evaluating data quality: {', '.join(selected_methods_data_quality)}")
    else:
        print("Selected imputation methods after evaluating data quality is None")
    if selected_methods_key_factors:
        print(f"Selected imputation methods after evaluating key factors: {', '.join(selected_methods_key_factors)}")
    else:
        print("Selected imputation methods after evaluating key factors is None")
    return selected_methods_final  # 返回所有符合条件的填补算法列表


# 主程序
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python selection_pruning.py <task_type> <dataset_file>")
        sys.exit(1)
    input_task_type = sys.argv[1]
    if input_task_type not in ["timeseries", "classification", "regression"]:
        print("Unsupported task type")
        sys.exit(1)
    input_missing_file = sys.argv[2]
    dataset = os.path.basename(os.path.dirname(os.path.dirname(input_missing_file)))
    if input_task_type == "timeseries" or input_task_type == "regression":
        dataset_info = datasets_numerical[dataset]
    elif input_task_type == "classification":
        dataset_info = datasets_categorical[dataset]

    input_missing_data = pd.read_csv(input_missing_file)
    target_column = dataset_info["target_column"]
    missing_rate = int(calculate_missing_rate(input_missing_data, target_column) * 100)
    print(f"Missing rate of the dataset '{dataset}': {missing_rate}%")

    performance_threshold = 0.9  # 性能阈值
    data_quality_threshold = {
        "2_wasserstein_distance": 1,
        "kl_divergence": 0.05,
        "ks_test": 0.05,
        "mutual_information": 2,
        "sliced_wasserstein_distance": 1.5
    }  # 数据质量阈值
    timeseries_key_factors_threshold = {
        "trend": 0.9,
        "seasonal": 0.9,
        "resid": 0.9
    }
    classification_key_factors_threshold = {
        "label_correctness_ratio": None,
        "class_discriminability": 0.2
    }
    regression_key_factors_threshold = {
        "imputation_deviation": None,
        "feature_target_corr": 0.85
    }  # 关键因素阈值

    if input_task_type == "timeseries":
        selected_imputation_methods = select_imputation_strategy(missing_rate, input_task_type, performance_threshold,
                                                                 data_quality_threshold,
                                                                 timeseries_key_factors_threshold)
    elif input_task_type == "classification":
        selected_imputation_methods = select_imputation_strategy(missing_rate, input_task_type, performance_threshold,
                                                                 data_quality_threshold,
                                                                 classification_key_factors_threshold)
    elif input_task_type == "regression":
        selected_imputation_methods = select_imputation_strategy(missing_rate, input_task_type, performance_threshold,
                                                                 data_quality_threshold,
                                                                 regression_key_factors_threshold)

    if selected_imputation_methods:
        print(f"Selected imputation methods: {', '.join(selected_imputation_methods)}")
    else:
        print("No suitable imputation methods found")