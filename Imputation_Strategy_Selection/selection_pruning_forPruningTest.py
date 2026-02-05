import numpy as np
import pandas as pd
import os
import sys
inf = 1000

datasets_timeseriesforecasting = {
    "ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},
    "ETTm1": {"target_column": "OT", "nonnumerical_column": "date"},
    "Illness": {"target_column": "OT", "nonnumerical_column": "date"},
    "Exchange": {"target_column": "OT", "nonnumerical_column": "date"},
    "Weather": {"target_column": "OT", "nonnumerical_column": "date"},
}

datasets_classification = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    "Flights": {"target_column": "flight", "unrelated_column": "None"},
    "Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"},
    "RedWineQuality": {"target_column": "quality", "unrelated_column": "None"},
    "AvocadoRipeness": {"target_column": "ripeness", "unrelated_column": "None"}
}

datasets_regression = {
    "concrete": {"target_column": "concrete_compressive_strength", "nonnumerical_column": "None"},
    "CCPP": {"target_column": "PE", "nonnumerical_column": "None"},
    "AirfoilSelfNoise": {"target_column": "SSPL", "nonnumerical_column": "None"},
    "Abalone": {"target_column": "Rings", "nonnumerical_column": "None"},
    "ParisHousing": {"target_column": "price", "nonnumerical_column": "None"},
}

test_configs = {
    "timeseries": {"ETTh2-test": {"target_column": "OT", "nonnumerical_column": "date"}},
    "classification": {"Glass-test": {"target_column": "Type", "unrelated_column": "None"}},
    "regression": {"BostonHousePrice-test": {"target_column": "MEDV", "nonnumerical_column": "None"}}
}

#test_task_types = ["timeseries", "classification", "regression"]
test_task_types = ["classification", "regression"]
Imputation_Algorithms_Numerical = ['mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'missfi', 'xgbi', 'gain', 'midae']
Imputation_Algorithms_Categorical = ['mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


# Search for downstream tasks that meet the requirements in the knowledge base (only one dataset is ok)
def evaluate_downstream_task_performance(missing_rate, input_task_type, sigma, dataset):
    base_path = "../Downstream_Results"
    history_dataset = dataset + "-history"
    downstream_task_performance_success = []
    if input_task_type == "timeseries":
        for model in Imputation_Algorithms_Numerical:
            history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
            #knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
            downstream_task_performance_file = pd.read_csv(os.path.join(base_path, input_task_type, history_dataset, f"mlp-imputation-results-{history_dataset}.csv"))
            #row = downstream_task_performance_file[downstream_task_performance_file['File Name'] == knowledge_imputed_file]
            row = downstream_task_performance_file[downstream_task_performance_file['File Name'] == history_imputed_file]
            value = row['PG(RMSE)'].values[0]
            if value < sigma:
                downstream_task_performance_success.append(model)


    elif input_task_type == "classification":
        for model in Imputation_Algorithms_Categorical:
            history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
            #knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
            downstream_task_performance_file = pd.read_csv(os.path.join(base_path, input_task_type, history_dataset, f"mlp-imputation-results-{history_dataset}.csv"))
            #row = downstream_task_performance_file[downstream_task_performance_file['File Name'] == knowledge_imputed_file]
            row = downstream_task_performance_file[downstream_task_performance_file['File Name'] == history_imputed_file]
            value = row['PG(F1 Score)'].values[0]
            if value < sigma:
                downstream_task_performance_success.append(model)


    elif input_task_type == "regression":
        for model in Imputation_Algorithms_Numerical:
            history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
            #knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
            downstream_task_performance_file = pd.read_csv(os.path.join(base_path, input_task_type, history_dataset, f"mlp-imputation-results-{history_dataset}.csv"))
            row = downstream_task_performance_file[downstream_task_performance_file['File Name'] == history_imputed_file]
            value = row['PG(MAE)'].values[0]
            if value < sigma:
                downstream_task_performance_success.append(model)
    return downstream_task_performance_success

# Search for the required upper/lower limits in the knowledge base
def evaluate_data_quality(missing_rate, input_task_type, sigma, dataset):
    performance_success = evaluate_downstream_task_performance(missing_rate, input_task_type, sigma, dataset)
    methods_data_quality_success_list = performance_success
    base_path = "../Datasets"
    wasserstein_distance_max = -inf# Find the maximum value that meets the requirements in the knowledge base
    kl_divergence_max = -inf# Find the maximum value that meets the requirements in the knowledge base
    ks_test_min = inf# Find the minimum value that meets the requirements in the knowledge base
    mutual_information_min = inf# Find the minimum value that meets the requirements in the knowledge base
    sliced_wasserstein_distance_max= -inf# Find the maximum value that meets the requirements in the knowledge base
    for model in performance_success:
        knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
        for dataset in {**datasets_timeseriesforecasting, **datasets_regression}:
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
                print("empty")
            value = row['avg_ratio'].values[0]
            if value > sliced_wasserstein_distance_max:
                sliced_wasserstein_distance_max = value
    return methods_data_quality_success_list, wasserstein_distance_max, kl_divergence_max, ks_test_min, mutual_information_min, sliced_wasserstein_distance_max

# Calculate the possible data quality metrics based on the historical data of the input data
def calculate_data_quality(missing_rate, input_task_type, model, dataset):
    #parts = input_missing_file.split('/')
    #history_dataset = parts[2]
    #default history dataset
    #history_dataset = "M3-Yearly-history"
    history_dataset = "BostonHousePrice-history"
    if input_task_type == "timeseries" or input_task_type == "regression":
        history_dataset = dataset + "-history"

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
    #KS classification_overview
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


# key factors of time series forecasting: trend, seasonality, residual
def evaluate_key_factors_timeseries(missing_rate, input_task_type, sigma, dataset):
    performance_success = evaluate_downstream_task_performance(missing_rate, input_task_type, sigma, dataset)
    methods_key_factors_timeseries_success_list = performance_success
    #print(f"Performance success: {performance_success}")
    base_path = "../Datasets"
    trend_min = inf
    seasonal_min =inf
    resid_min = inf
    for model in performance_success:
        knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
        for dataset in datasets_timeseriesforecasting:
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
    #print(f"Trend min: {trend_min}", f"seasonal min: {seasonal_min}", f"resid min: {resid_min}")
    return methods_key_factors_timeseries_success_list, trend_min, seasonal_min, resid_min

def calculate_key_factors_timeseries(missing_rate, model, dataset):
    #parts = input_missing_file.split('/')
    #history_dataset = parts[2]
    history_dataset = dataset + "-history"
    history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
    base_path = "../Datasets"
    key_factors_file = pd.read_csv(os.path.join(base_path, history_dataset, "Mechanism", "timeseries", "decompose_basic", "decompose_basic_results.csv"))
    row = key_factors_file[key_factors_file['file'] == history_imputed_file]
    trend_value = row['Corr_Trend'].values[0]
    seasonal_value = row['Corr_Seasonal'].values[0]
    resid_value = row['Corr_Residuals'].values[0]
    return trend_value, seasonal_value, resid_value

def evaluate_key_factors_classification(missing_rate, input_task_type, sigma, dataset):
    performance_success = evaluate_downstream_task_performance(missing_rate, input_task_type, sigma, dataset)
    methods_key_factors_classification_success_list = performance_success
    base_path = "../Datasets"
    label_correctness_ratio_min = inf
    class_discriminability_min = inf
    for model in performance_success:
        knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
        for dataset in datasets_classification:
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
    return methods_key_factors_classification_success_list, label_correctness_ratio_min, class_discriminability_min

def calculate_key_factors_classification(missing_rate, model, dataset):
    #parts = input_missing_file.split('/')
    #history_dataset = parts[2]
    history_dataset = dataset + "-history"
    history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
    base_path = "../Datasets"
    key_factors_file = pd.read_csv(os.path.join(base_path, history_dataset, "Mechanism", "classification", f"label_correctness_ratio_results.csv"))
    row = key_factors_file[key_factors_file['file'] == history_imputed_file]
    label_correctness_ratio_value = row['Consistent Rate'].values[0]

    key_factors_file = pd.read_csv(os.path.join(base_path, history_dataset, "Mechanism", "classification", f"class_discriminability_results.csv"))
    row = key_factors_file[key_factors_file['file'] == history_imputed_file]
    class_discriminability_value = row['J Value'].values[0]
    return label_correctness_ratio_value, class_discriminability_value

def evaluate_key_factors_regression(missing_rate, input_task_type, sigma, dataset):
    performance_success = evaluate_downstream_task_performance(missing_rate, input_task_type, sigma, dataset)
    methods_key_factors_regression_success_list = performance_success
    base_path = "../Datasets"
    imputation_deviation_max = -inf
    feature_target_corr_min = inf
    for model in performance_success:
        knowledge_imputed_file = f"dirty-{model}-{missing_rate}.csv"
        for dataset in datasets_regression:
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
    return methods_key_factors_regression_success_list, imputation_deviation_max, feature_target_corr_min

def calculate_key_factors_regression(missing_rate, model, dataset):
    # = input_missing_file.split('/')
    #history_dataset = parts[2]
    history_dataset = dataset + "-history"
    history_imputed_file = f"dirty-{model}-{missing_rate}.csv"
    base_path = "../Datasets"
    key_factors_file = pd.read_csv(os.path.join(base_path, history_dataset, "Mechanism", "regression", "imputation_deviation", f"imputation_deviation_results.csv"))
    row = key_factors_file[key_factors_file['file'] == history_imputed_file]
    imputation_deviation_value = row['MAE'].values[0]
    key_factors_file = pd.read_csv(os.path.join(base_path, history_dataset, "Mechanism", "regression", "feature_target_corr", f"feature_target_corr_results.csv"))
    row = key_factors_file[key_factors_file['file'] == history_imputed_file]
    feature_target_corr_value = row['Avg_Correlation'].values[0]
    return imputation_deviation_value, feature_target_corr_value


# 定义填补策略选择函数
def select_imputation_strategy(missing_rate, input_task_type, sigma1, sigma2, sigma3, dataset):
    if input_task_type == "timeseries" or input_task_type == "regression":
        Alternative_algorithms = Imputation_Algorithms_Numerical
    elif input_task_type == "classification":
        Alternative_algorithms = Imputation_Algorithms_Categorical
    selected_methods_performance = []
    selected_methods_data_quality = []
    selected_methods_key_factors = []
    selected_methods_final = []
    print("Establishing mapping relationships between the knowledge base and sigma1/sigma2/sigma3...")
    print("Calculating downstream task performance...")
    methods_performance_success_list = evaluate_downstream_task_performance(missing_rate, input_task_type, sigma1, dataset)
    print("Calculating data quality...")
    methods_data_quality_success_list, wasserstein_distance_max, kl_divergence_max, ks_test_min, mutual_information_min, sliced_wasserstein_distance_max = evaluate_data_quality(missing_rate, input_task_type, sigma2, dataset)
    #print(wasserstein_distance_max, kl_divergence_max, ks_test_min, mutual_information_min, sliced_wasserstein_distance_max)
    print("Calculating key factors...")
    if input_task_type == "timeseries":
        methods_key_factors_timeseries_success_list, trend_min, seasonal_min, resid_min = evaluate_key_factors_timeseries(missing_rate, input_task_type, sigma3, dataset)
    elif input_task_type == "classification":
        methods_key_factors_classification_success_list, label_correctness_ratio_min, class_discriminability_min = evaluate_key_factors_classification(missing_rate, input_task_type, sigma3, dataset)
    elif input_task_type == "regression":
        methods_key_factors_regression_success_list, imputation_deviation_max, feature_target_corr_min = evaluate_key_factors_regression(missing_rate, input_task_type, sigma3, dataset)


    for model in Alternative_algorithms:
        # Performance Evaluation
        if model not in methods_performance_success_list:
            continue
        else:
            #print(f"{model}通过性能评估")
            selected_methods_performance.append(model)

        # Data Quality Evaluation
        wasserstein_distance_value, kl_divergence_value, ks_test_value, mutual_information_value, sliced_wasserstein_distance_value = calculate_data_quality(missing_rate, input_task_type, model, dataset)
        if model not in methods_data_quality_success_list:
            continue
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
            #print(f"{model} pass the data quality evaluation")
            selected_methods_data_quality.append(model)

        # Key Factors Analysis
        if input_task_type == "timeseries":
            if model not in methods_key_factors_timeseries_success_list:
                continue
            trend_value, seasonal_value, resid_value = calculate_key_factors_timeseries(missing_rate, model, dataset)
            if trend_value < trend_min:
                continue
            elif seasonal_value <seasonal_min:
                continue
            elif resid_value < resid_min:
                continue
            else:
                #print(f"{model} pass the key factors analysis")
                selected_methods_key_factors.append(model)

        elif input_task_type == "classification":
            if model not in methods_key_factors_classification_success_list:
                continue
            label_correctness_ratio_value, class_discriminability_value = calculate_key_factors_classification(missing_rate, model, dataset)
            if label_correctness_ratio_value < label_correctness_ratio_min:
                continue
            elif class_discriminability_value < class_discriminability_min:
                continue
            else:
                #print(f"{model} pass the key factors analysis")
                selected_methods_key_factors.append(model)

        elif input_task_type == "regression":
            if model not in methods_key_factors_regression_success_list:
                continue
            imputation_deviation_value, feature_target_corr_value = calculate_key_factors_regression(missing_rate, model, dataset)
            if imputation_deviation_value > imputation_deviation_max:
                continue
            elif feature_target_corr_value < feature_target_corr_min:
                continue
            else:
                #print(f"{model} pass the key factors analysis")
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
    print("="*100)
    print(f"-------------------{dataset}-{missing_rate}-{input_task_type}-------------------")
    return {
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma3": sigma3,
        "selected_methods_performance": selected_methods_performance,
        "selected_methods_data_quality": selected_methods_data_quality,
        "selected_methods_key_factors": selected_methods_key_factors,
        "selected_methods_final": selected_methods_final
    }


if __name__ == "__main__":
    for input_task_type in test_task_types:
        datasets_config = test_configs.get(input_task_type, {})
        for dataset, dataset_info in datasets_config.items():
            dataset = dataset.replace("-test", "")
            target_column = dataset_info["target_column"]
            for missing_rate in Missing_rate:
                input_missing_file = os.path.join("../Datasets", dataset, "null", f"dirty-{missing_rate}.csv")
                input_missing_data = pd.read_csv(input_missing_file)
                print(f"Missing rate of the dataset '{dataset}-dirty-{missing_rate}': {missing_rate}%")

                results = []

                for sigma1 in np.arange(0.05, 1.0, 0.05):
                #for sigma1 in np.arange(0.95, 0.95+0.0001, 0.05):
                    for sigma2 in np.arange(sigma1 - 0.05, sigma1 + 0.0001, 0.01):
                        for sigma3 in np.arange(max(sigma2 - 0.05, 0.01), sigma2 + 0.0001, 0.01):
                            sigma1 = round(sigma1, 2)
                            sigma2 = round(sigma2, 2)
                            sigma3 = round(sigma3, 2)
                            print(sigma1, sigma2, sigma3)
                            result = select_imputation_strategy(missing_rate, input_task_type, sigma1, sigma2, sigma3,
                                                                dataset)
                            results.append(result)

                # output
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
                        f.write("=" * 100 + "\n")

                print(f"Results have been saved to {output_file}")