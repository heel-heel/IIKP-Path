import numpy as np
import pandas as pd
import os
import sys
inf = 1000

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
Imputation_Algorithms_Numerical = ['mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'missfi', 'xgbi', 'gain', 'midae']
Imputation_Algorithms_Categorical = ['mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'missfi', 'xgbi', 'gain', 'midae']


# 定义计算目标列缺失率的函数
def calculate_missing_rate(input_missing_data, target_column):
    total_cells = input_missing_data[target_column].size  # 目标列的总单元格数
    missing_cells = input_missing_data[target_column].isnull().sum()  # 目标列的缺失单元格数
    missing_rate = missing_cells / total_cells  # 目标列的缺失率
    return missing_rate

#在知识库中查找符合要求的下游分析任务（只要某一个数据集即可）
def evaluate_downstream_task_performance(missing_rate, input_task_type, sigma):
    base_path = "../Downstream_Results"
    downstream_task_performance_success = []
    if input_task_type == "timeseries":
        for model in Imputation_Algorithms_Numerical:
            knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
            flag = 1
            for dataset in datasets_numerical:
                downstream_task_performance_file = pd.read_csv(os.path.join(base_path, input_task_type, dataset, f"mlp-imputation-results-{dataset}.csv"))
                row = downstream_task_performance_file[downstream_task_performance_file['File Name'] == knowledge_imputed_file]
                value = row['PG(RMSE)'].values[0]
                #print(f"{model}:{value}")
                if value > sigma:
                    flag = 0
                    #downstream_task_performance_success.append(model)
                    break
            if flag == 1:
                downstream_task_performance_success.append(model)

    elif input_task_type == "classification":
        for model in Imputation_Algorithms_Categorical:
            knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
            flag = 1
            for dataset in datasets_categorical:
                downstream_task_performance_file = pd.read_csv(os.path.join(base_path, input_task_type, dataset, f"mlp-imputation-results-{dataset}.csv"))
                row = downstream_task_performance_file[downstream_task_performance_file['File Name'] == knowledge_imputed_file]
                value = row['PG(F1 Score)'].values[0]
                #print(f"{model}:{value}")
                if value > sigma:
                    flag = 0
                    break
            if flag == 1:
                downstream_task_performance_success.append(model)
    elif input_task_type == "regression":
        for model in Imputation_Algorithms_Numerical:
            knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
            flag = 1
            for dataset in datasets_numerical:
                downstream_task_performance_file = pd.read_csv(os.path.join(base_path, input_task_type, dataset, f"mlp-imputation-results-{dataset}.csv"))
                row = downstream_task_performance_file[downstream_task_performance_file['File Name'] == knowledge_imputed_file]
                value = row['PG(MAE)'].values[0]
                #print(f"{model}:{value}")
                if value > sigma:
                    flag = 0
                    break
            if flag == 1:
                downstream_task_performance_success.append(model)
    return downstream_task_performance_success

# 在知识库中查找符合要求的上限/下限
def evaluate_data_quality(missing_rate, input_task_type, sigma):
    performance_success = evaluate_downstream_task_performance(missing_rate, input_task_type, sigma)
    base_path = "../Datasets"
    wasserstein_distance_max = -inf#求知识库里能符合要求的最大值
    kl_divergence_max = -inf#求知识库里能符合要求的最大值
    ks_test_min = inf#求知识库里能符合要求的最小值
    mutual_information_min = inf#求知识库里能符合要求的最小值
    sliced_wasserstein_distance_max= -inf#求知识库里能符合要求的最大值
    for model in performance_success:
        knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
        for dataset in datasets_numerical:
            data_quality_file = pd.read_csv(os.path.join(base_path, dataset, "Data_Quality", "2_wasserstein_distance", "2_wasserstein_distance_results.csv"))
            row = data_quality_file[data_quality_file['file'] == knowledge_imputed_file]
            value = row['2-Wasserstein Distance'].values[0]
            if value > wasserstein_distance_max:
                wasserstein_distance_max = value

            data_quality_file = pd.read_csv(os.path.join(base_path, dataset, "Data_Quality", "kl_divergence", "kl_divergence_results.csv"))
            row = data_quality_file[data_quality_file['file'] == knowledge_imputed_file]
            value = row['KL_Divergence'].values[0]
            if value > kl_divergence_max:
                kl_divergence_max = value

            data_quality_file = pd.read_csv(os.path.join(base_path, dataset, "Data_Quality", "ks_test", "ks_test_results.csv"))
            row = data_quality_file[data_quality_file['file'] == knowledge_imputed_file]
            value = row['P-Value'].values[0]
            if value < ks_test_min:
                ks_test_min = value

            data_quality_file = pd.read_csv(os.path.join(base_path, dataset, "Data_Quality", "mutual_information", "mutual_information_results.csv"))
            row = data_quality_file[data_quality_file['file'] == knowledge_imputed_file]
            value = row['Mutual_Information_target'].values[0]
            if value < mutual_information_min:
                mutual_information_min = value

            data_quality_file = pd.read_csv(os.path.join(base_path, dataset, "Data_Quality", "sliced_wasserstein_distance", "sliced_wasserstein_distance_results.csv"))
            row = data_quality_file[data_quality_file['file'] == knowledge_imputed_file]
            if row.empty:
                print("11111111111")
            value = row['avg_ratio'].values[0]
            if value > sliced_wasserstein_distance_max:
                sliced_wasserstein_distance_max = value
    return wasserstein_distance_max, kl_divergence_max, ks_test_min, mutual_information_min, sliced_wasserstein_distance_max

#根据输入数据的历史数据，计算可能的数据质量指标
def calculate_data_quality(missing_rate, model, input_missing_file):
    parts = input_missing_file.split('/')
    #history_dataset = parts[2]
    history_dataset = "M4-Monthly"
    history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
    base_path = "../Datasets"
    #2-wasserstein distance
    data_quality_file = pd.read_csv(os.path.join(base_path, history_dataset, "Data_Quality", "2_wasserstein_distance", "2_wasserstein_distance_results.csv"))
    row = data_quality_file[data_quality_file['file'] == history_imputed_file]
    wasserstein_distance_value = row['2-Wasserstein Distance'].values[0]
    #KL divergence
    data_quality_file = pd.read_csv(os.path.join(base_path, history_dataset, "Data_Quality", "kl_divergence", "kl_divergence_results.csv"))
    row = data_quality_file[data_quality_file['file'] == history_imputed_file]
    kl_divergence_value = row['KL_Divergence'].values[0]
    #KS test
    data_quality_file = pd.read_csv(os.path.join(base_path, history_dataset, "Data_Quality", "ks_test", "ks_test_results.csv"))
    row = data_quality_file[data_quality_file['file'] == history_imputed_file]
    ks_test_value = row['P-Value'].values[0]
    #mutual information
    data_quality_file = pd.read_csv(os.path.join(base_path, history_dataset, "Data_Quality", "mutual_information", "mutual_information_results.csv"))
    row = data_quality_file[data_quality_file['file'] == history_imputed_file]
    mutual_information_value = row['Mutual_Information_target'].values[0]
    #sliced wasserstein distance
    data_quality_file = pd.read_csv(os.path.join(base_path, history_dataset, "Data_Quality", "sliced_wasserstein_distance", "sliced_wasserstein_distance_results.csv"))
    row = data_quality_file[data_quality_file['file'] == history_imputed_file]
    sliced_wasserstein_distance_value = row['avg_ratio'].values[0]
    return wasserstein_distance_value, kl_divergence_value, ks_test_value, mutual_information_value, sliced_wasserstein_distance_value


# 时间序列预测的关键因素：趋势、残差、季节性
def evaluate_key_factors_timeseries(missing_rate, input_task_type, sigma):
    performance_success = evaluate_downstream_task_performance(missing_rate, input_task_type, sigma)
    base_path = "../Datasets"
    trend_min = inf
    seasonal_min =inf
    resid_min = inf
    for model in performance_success:
        knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
        for dataset in datasets_numerical:
            key_factors_file = pd.read_csv(os.path.join(base_path, dataset, "Mechanism", "timeseries", "decompose_basic", "decompose_basic_results.csv"))
            row = key_factors_file[key_factors_file['file'] == knowledge_imputed_file]
            value = row['Corr_Trend'].values[0]
            if value < trend_min:
                trend_min = value
            value = row['Corr_Seasonal'].values[0]
            if value < seasonal_min:
                seasonal_min = value
            value = row['Corr_Residuals'].values[0]
            if value < resid_min:
                resid_min = value
    return trend_min, seasonal_min, resid_min

def calculate_key_factors_timeseries(missing_rate, model, input_missing_file):
    parts = input_missing_file.split('/')
    history_dataset = parts[2]
    history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
    base_path = "../Datasets"
    key_factors_file = pd.read_csv(os.path.join(base_path, history_dataset, "Mechanism", "timeseries", "decompose_basic", "decompose_basic_results.csv"))
    row = key_factors_file[key_factors_file['file'] == history_imputed_file]
    trend_value = row['Corr_Trend'].values[0]
    seasonal_value = row['Corr_Seasonal'].values[0]
    resid_value = row['Corr_Residuals'].values[0]
    return trend_value, seasonal_value, resid_value

def evaluate_key_factors_classification(missing_rate, input_task_type, sigma):
    performance_success = evaluate_downstream_task_performance(missing_rate, input_task_type, sigma)
    base_path = "../Datasets"
    label_correctness_ratio_min = inf
    class_discriminability_min = inf
    for model in performance_success:
        knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
        for dataset in datasets_categorical:
            key_factors_file = pd.read_csv(os.path.join(base_path, dataset, "Mechanism", "classification", f"label_correctness_ratio_results.csv"))
            row = key_factors_file[key_factors_file['file'] == knowledge_imputed_file]
            value = row['Consistent Rate'].values[0]
            if value < label_correctness_ratio_min:
                label_correctness_ratio_min= value

            key_factors_file = pd.read_csv(os.path.join(base_path, dataset, "Mechanism", "classification", f"class_discriminability_results.csv"))
            row = key_factors_file[key_factors_file['file'] == knowledge_imputed_file]
            value = row['J Value'].values[0]
            if value < class_discriminability_min:
                class_discriminability_min = value
    return label_correctness_ratio_min, class_discriminability_min

def calculate_key_factors_classification(missing_rate, model, input_missing_file):
    parts = input_missing_file.split('/')
    history_dataset = parts[2]
    history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
    base_path = "../Datasets"
    key_factors_file = pd.read_csv(os.path.join(base_path, history_dataset, "Mechanism", "classification", f"label_correctness_ratio_results.csv"))
    row = key_factors_file[key_factors_file['file'] == history_imputed_file]
    label_correctness_ratio_value = row['Consistent Rate'].values[0]

    key_factors_file = pd.read_csv(os.path.join(base_path, history_dataset, "Mechanism", "classification", f"class_discriminability_results.csv"))
    row = key_factors_file[key_factors_file['file'] == history_imputed_file]
    class_discriminability_value = row['J Value'].values[0]
    return label_correctness_ratio_value, class_discriminability_value

def evaluate_key_factors_regression(missing_rate, input_task_type, sigma):
    performance_success = evaluate_downstream_task_performance(missing_rate, input_task_type, sigma)
    base_path = "../Datasets"
    imputation_deviation_max = -inf
    feature_target_corr_min = inf
    for model in performance_success:
        knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
        for dataset in datasets_numerical:
            key_factors_file = pd.read_csv(os.path.join(base_path, dataset, "Mechanism", "regression", "imputation_deviation", f"imputation_deviation_results.csv"))
            row = key_factors_file[key_factors_file['file'] == knowledge_imputed_file]
            value = row['MAE'].values[0]
            if value > imputation_deviation_max:
                imputation_deviation_max = value

            key_factors_file = pd.read_csv(os.path.join(base_path, dataset, "Mechanism", "regression", "feature_target_corr", f"feature_target_corr_results.csv"))
            row = key_factors_file[key_factors_file['file'] == knowledge_imputed_file]
            value = row['Avg_Correlation'].values[0]
            if value < feature_target_corr_min:
                feature_target_corr_min = value
    return imputation_deviation_max, feature_target_corr_min

def calculate_key_factors_regression(missing_rate, model, input_missing_file):
    parts = input_missing_file.split('/')
    history_dataset = parts[2]
    history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
    base_path = "../Datasets"
    key_factors_file = pd.read_csv(os.path.join(base_path, dataset, "Mechanism", "regression", "imputation_deviation", f"imputation_deviation_results.csv"))
    row = key_factors_file[key_factors_file['file'] == history_imputed_file]
    imputation_deviation_value = row['MAE'].values[0]
    key_factors_file = pd.read_csv(os.path.join(base_path, dataset, "Mechanism", "regression", "feature_target_corr", f"feature_target_corr_results.csv"))
    row = key_factors_file[key_factors_file['file'] == history_imputed_file]
    feature_target_corr_value = row['Avg_Correlation'].values[0]
    return imputation_deviation_value, feature_target_corr_value


# 定义填补策略选择函数
def select_imputation_strategy(missing_rate, input_task_type, sigma1, sigma2, sigma3, input_missing_file):
    if input_task_type == "timeseries" or input_task_type == "regression":
        Alternative_algorithms = Imputation_Algorithms_Numerical
    elif input_task_type == "classification":
        Alternative_algorithms = Imputation_Algorithms_Categorical
    selected_methods_performance = []
    selected_methods_data_quality = []
    selected_methods_key_factors = []
    selected_methods_final = []
    print("正在将知识库与sigma1/sigma2/sigma3建立映射关系...")
    print("正在计算分析任务性能...")
    metheds_performance_success_list = evaluate_downstream_task_performance(missing_rate, input_task_type, sigma1)
    print("正在计算数据质量...")
    wasserstein_distance_max, kl_divergence_max, ks_test_min, mutual_information_min, sliced_wasserstein_distance_max = evaluate_data_quality(missing_rate, input_task_type, sigma2)
    #print(wasserstein_distance_max, kl_divergence_max, ks_test_min, mutual_information_min, sliced_wasserstein_distance_max)
    print("正在计算关键因素...")
    if input_task_type == "timeseries":
        trend_min, seasonal_min, resid_min = evaluate_key_factors_timeseries(missing_rate, input_task_type, sigma3)
    elif input_task_type == "classification":
        label_correctness_ratio_min, class_discriminability_min = evaluate_key_factors_classification(missing_rate, input_task_type, sigma3)
    elif input_task_type == "regression":
        imputation_deviation_max, feature_target_corr_min = evaluate_key_factors_regression(missing_rate, input_task_type, sigma3)

    for model in Alternative_algorithms:
        #性能评估
        if model not in metheds_performance_success_list:
            continue
        else:
            #print(f"{model}通过性能评估")
            selected_methods_performance.append(model)

        #数据质量评估
        wasserstein_distance_value, kl_divergence_value, ks_test_value, mutual_information_value, sliced_wasserstein_distance_value = calculate_data_quality(missing_rate, model, input_missing_file)
        if wasserstein_distance_value > wasserstein_distance_max:
            continue
        elif kl_divergence_value > kl_divergence_max:
            continue
        elif ks_test_value < ks_test_min:
            continue
        elif mutual_information_value < mutual_information_min:
            continue
        elif sliced_wasserstein_distance_value > sliced_wasserstein_distance_max:
            continue
        else:
            #print(f"{model}通过数据质量评估")
            selected_methods_data_quality.append(model)

        #关键因素评估
        if input_task_type == "timeseries":
            trend_value, seasonal_value, resid_value = calculate_key_factors_timeseries(missing_rate, model, input_missing_file)
            if trend_value < trend_min:
                continue
            elif seasonal_value <seasonal_min:
                continue
            elif resid_value < resid_min:
                continue
            else:
                #print(f"{model}通过关键因素评估")
                selected_methods_key_factors.append(model)

        elif input_task_type == "classification":
            label_correctness_ratio_value, class_discriminability_value = calculate_key_factors_classification(missing_rate, model, input_missing_file)
            if label_correctness_ratio_value < label_correctness_ratio_min:
                continue
            elif class_discriminability_value < class_discriminability_min:
                continue
            else:
                #print(f"{model}通过关键因素评估")
                selected_methods_key_factors.append(model)

        elif input_task_type == "regression":
            imputation_deviation_value, feature_target_corr_value = calculate_key_factors_regression(missing_rate, model, input_missing_file)
            if imputation_deviation_value > imputation_deviation_max:
                continue
            elif feature_target_corr_value < feature_target_corr_min:
                continue
            else:
                #print(f"{model}通过关键因素评估")
                selected_methods_key_factors.append(model)
        selected_methods_final.append(model)
    if selected_methods_performance:
        print(f"Selected imputation methods after evaluating downstream task's performance: {', '.join(selected_methods_performance)}")
    else:
        print("Selected imputation methods after evaluating downstream task's performance is None")
    if selected_methods_data_quality:
        print(f"Selected imputation methods after evaluating data quality: {', '.join(selected_methods_data_quality)}")
    else:
        print("Selected imputation methods after evaluating data quality is None")
    if selected_methods_key_factors:
        print(f"Selected imputation methods after evaluating key factors: {', '.join(selected_methods_key_factors)}")
    else:
        print("Selected imputation methods after evaluating key factors is None")
    print("="*70)
    return {
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma3": sigma3,
        "selected_methods_performance": selected_methods_performance,
        "selected_methods_data_quality": selected_methods_data_quality,
        "selected_methods_key_factors": selected_methods_key_factors,
        "selected_methods_final": selected_methods_final
    }


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
    missing_rate = round(calculate_missing_rate(input_missing_data, target_column) * 100)
    print(f"Missing rate of the dataset '{dataset}': {missing_rate}%")

    '''
    sigma1 = 0.1
    sigma2 = 0.05
    sigma3 = 0.03

    #input_missing_data用来处理历史数据
    selected_imputation_methods = select_imputation_strategy(missing_rate, input_task_type, sigma1, sigma2, sigma3, input_missing_file)

    if selected_imputation_methods:
        print(f"Selected imputation methods: {', '.join(selected_imputation_methods)}")
    else:
        print("No suitable imputation methods found")
    '''

    results = []

    for sigma1 in np.arange(0.05, 0.55, 0.05):
        for sigma2 in np.arange(sigma1 - 0.05, sigma1, 0.01):
            for sigma3 in np.arange(max(sigma2 - 0.05, 0.01), sigma2, 0.01):
                sigma1 = round(sigma1, 2)
                sigma2 = round(sigma2, 2)
                sigma3 = round(sigma3, 2)
                print(sigma1, sigma2, sigma3)
                result = select_imputation_strategy(missing_rate, input_task_type, sigma1, sigma2, sigma3,
                                                    input_missing_file)
                results.append(result)

    # 导出结果到文件
    output_dir = f"./Results/{input_task_type}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{dataset}_{missing_rate}_results.txt"

    with open(output_file, 'w') as f:
        for result in results:
            f.write(f"Sigma1: {result['sigma1']}, Sigma2: {result['sigma2']}, Sigma3: {result['sigma3']}\n")
            f.write(
                f"Selected methods after evaluating downstream task's performance: {', '.join(result['selected_methods_performance'])}\n")
            f.write(
                f"Selected methods after evaluating data quality: {', '.join(result['selected_methods_data_quality'])}\n")
            f.write(f"Selected methods after evaluating key factors: {', '.join(result['selected_methods_key_factors'])}\n")
            f.write(f"Final selected methods: {', '.join(result['selected_methods_final'])}\n")
            f.write("=" * 70 + "\n")

    print(f"Results have been saved to {output_file}")