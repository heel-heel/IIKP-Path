import os
import numpy as np
import pandas as pd

test_configs = {
    "timeseries": {"M3-Yearly-history": {"target_column": "V2", "nonnumerical_column": "V1"}},
    "classification": {"Glass-history": {"target_column": "Type", "unrelated_column": "None"}},
    "regression": {"BostonHousePrice-history": {"target_column": "MEDV", "nonnumerical_column": "None"}}
}
test_task_types = ["timeseries", "classification", "regression"]
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

def check_imputation_algorithms(input_task_type, dataset, missing_rate, sigma1, results):
    downstream_task_performance_file = pd.read_csv(os.path.join("../Downstream_Results", f"{input_task_type}", dataset, f"mlp-imputation-results-{dataset}.csv"))

    # 根据任务类型确定要检查的性能指标列
    if input_task_type == "timeseries":
        target_metric = "PG(RMSE)"
    elif input_task_type == "classification":
        target_metric = "PG(F1 Score)"
    elif input_task_type == "regression":
        target_metric = "PG(MAE)"

    # 筛选包含指定缺失率的行
    matching_rows = downstream_task_performance_file[downstream_task_performance_file['File Name'].str.contains(f'-{missing_rate}.csv')]

    success_model_list = []

    # 检查性能指标是否在sigma1以下
    for _, row in matching_rows.iterrows():
        file_name = row['File Name']
        file_name_parts = file_name[6:-4].split('-')
        if len(file_name_parts) == 2:
            model = ''.join(file_name_parts[:-1])
        else:
            continue
        performance_value = row[target_metric]

        if performance_value <= sigma1:
            success_model_list.append(model)
    return success_model_list

# 主循环
for input_task_type in test_task_types:
    if input_task_type in test_configs:
        datasets_config = test_configs.get(input_task_type, {})
        for dataset, dataset_info in datasets_config.items():
            results = []
            for missing_rate in Missing_rate:
                for sigma1 in np.arange(0.05, 1.0, 0.05):
                    sigma1 = round(sigma1, 2)
                    success_model_list = check_imputation_algorithms(input_task_type, dataset, missing_rate, sigma1, results)
                    results.append({
                        'missing_rate': missing_rate,
                        'sigma1': sigma1,
                        'imputation methods': success_model_list,
                    })

            # 为每个数据集导出结果到txt文件
            output_dir = f"./Results/{input_task_type}"
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/{dataset}_actual_results.txt"
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(f"missing_rate: {result['missing_rate']}\n")
                    f.write(f"sigma1: {result['sigma1']}\n")
                    f.write(f"imputation methods: {result['imputation methods']}\n")
                    f.write("=" * 100 + "\n")

            print(f"Results have been saved to {output_file}")