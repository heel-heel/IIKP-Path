import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr


output_path = os.path.join("./factors_relation_results")
if not os.path.exists(output_path):
    os.makedirs(output_path)
output_file = os.path.join(output_path, "factors_relation_results.txt")

datasets = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    "Flights": {"target_column": "flight", "unrelated_column": None},
    "Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"},
    "RedWineQuality": {"target_column": "quality", "unrelated_column": "None"},
    "AvocadoRipeness": {"target_column": "ripeness", "unrelated_column": "None"}
}

with open(output_file, 'w') as f:
    for dataset, info in datasets.items():
        input_class_discriminability_path = os.path.join("../../Datasets", dataset, "Mechanism", "classification", "class_discriminability_results.csv")
        input_label_correctness_radio_path = os.path.join("../../Datasets", dataset, "Mechanism", "classification", "label_correctness_ratio_results.csv")
        input_pg_path = os.path.join("../../Downstream_Results", "classification", dataset, f"mlp-imputation-results-{dataset}.csv")

        input_class_discriminability_file = pd.read_csv(input_class_discriminability_path)
        input_label_correctness_radio_file = pd.read_csv(input_label_correctness_radio_path)
        input_pg_file = pd.read_csv(input_pg_path)
        input_pg_file.rename(columns={'File Name': 'file'}, inplace=True)
        merged_data = pd.merge(input_class_discriminability_file, input_label_correctness_radio_file, on='file', how='inner')
        merged_data = pd.merge(merged_data, input_pg_file, on='file', how='inner')
        merged_data_imputed = merged_data.dropna(subset=['J Value', 'Consistent Rate', 'PG(F1 Score)'])

        y = merged_data_imputed['PG(F1 Score)']
        X = merged_data_imputed[['J Value', 'Consistent Rate']]
        X = sm.add_constant(X)

        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        f.write(f"共线性检查结果（数据集：{dataset}）：\n")
        f.write(vif_data.to_string(index=False) + "\n")
        f.write("\n" + "-" * 70 + "\n")

        for independent_vars in [['J Value'], ['Consistent Rate'], ['J Value', 'Consistent Rate']]:
            X = merged_data_imputed[independent_vars]
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            f.write(f"回归分析结果（数据集：{dataset}，自变量：{', '.join(independent_vars)}）：\n")
            f.write(model.summary().as_text() + "\n")
            f.write("\n" + "-" * 70 + "\n")

        plt.figure(figsize=(14, 8))
        plt.scatter(merged_data_imputed['J Value'], merged_data_imputed['Consistent Rate'], alpha=0.7)
        plt.title(f'Scatter Plot of J Value vs Consistent Rate ({dataset})', fontsize=16)
        plt.xlabel('J Value', fontsize=14)
        plt.ylabel('Consistent Rate', fontsize=14)
        output_fig_path = os.path.join(output_path, "fig")
        if not os.path.exists(output_fig_path):
            os.makedirs(output_fig_path)
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, f"Scatter Plot of J Value vs Consistent Rate ({dataset}).png"))
        #plt.show()

        spearman_corr, p_value = spearmanr(merged_data_imputed['J Value'], merged_data_imputed['Consistent Rate'])
        f.write(f"Spearman 相关系数（数据集：{dataset}）：\n")
        f.write(f"相关系数: {spearman_corr:.4f}\n")
        f.write(f"P 值: {p_value:.4f}\n")
        f.write("\n" + "-" * 70 + "\n")