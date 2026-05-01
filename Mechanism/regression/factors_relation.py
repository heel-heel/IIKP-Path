import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr

downstream_model = 'cnn'
output_path = os.path.join("./factors_relation_results")
if not os.path.exists(output_path):
    os.makedirs(output_path)
output_file = os.path.join(output_path, f"factors_relation_results_{downstream_model}.txt")

datasets = {
    "concrete": {"target_column": "concrete_compressive_strength", "nonnumerical_column": "None"},
    "CCPP": {"target_column": "PE", "nonnumerical_column": "None"},
    "AirfoilSelfNoise": {"target_column": "SSPL", "nonnumerical_column": "None"},
    "Abalone": {"target_column": "Rings", "nonnumerical_column": "None"},
    "ParisHousing": {"target_column": "price", "nonnumerical_column": "None"},
}
Missing_Mechanism = 'MCAR'

with open(output_file, 'w') as f:
    for dataset, info in datasets.items():
        input_feature_target_corr_path = os.path.join("../../Datasets", dataset, "Mechanism", "regression", "feature_target_corr", "feature_target_corr_results.csv")
        input_imputation_deviation_path = os.path.join("../../Datasets", dataset, "Mechanism", "regression", "imputation_deviation", "imputation_deviation_results.csv")
        input_pg_path = os.path.join("../../Downstream_Results", "regression", dataset, Missing_Mechanism, f"{downstream_model}-imputation-results-{dataset}.csv")

        input_class_discriminability_file = pd.read_csv(input_feature_target_corr_path)
        input_label_correctness_radio_file = pd.read_csv(input_imputation_deviation_path)
        input_pg_file = pd.read_csv(input_pg_path)
        input_pg_file.rename(columns={'File Name': 'file'}, inplace=True)
        merged_data = pd.merge(input_class_discriminability_file, input_label_correctness_radio_file, on='file', how='inner')
        merged_data = pd.merge(merged_data, input_pg_file, on='file', how='inner')
        merged_data_imputed = merged_data.dropna(subset=['Avg_Correlation', 'RMSE', 'PG(MAE)'])

        y = merged_data_imputed['PG(MAE)']
        X = merged_data_imputed[['Avg_Correlation', 'RMSE']]
        X = sm.add_constant(X)

        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        f.write(f"共线性检查结果（数据集：{dataset}）：\n")
        f.write(vif_data.to_string(index=False) + "\n")
        f.write("\n" + "-" * 70 + "\n")

        for independent_vars in [['Avg_Correlation'], ['RMSE'], ['Avg_Correlation', 'RMSE']]:
            X = merged_data_imputed[independent_vars]
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            f.write(f"回归分析结果（数据集：{dataset}，自变量：{', '.join(independent_vars)}）：\n")
            f.write(model.summary().as_text() + "\n")
            f.write("\n" + "-" * 70 + "\n")

        plt.figure(figsize=(14, 8))
        plt.scatter(merged_data_imputed['Avg_Correlation'], merged_data_imputed['RMSE'], alpha=0.7)
        plt.title(f'Scatter Plot of Avg_Correlation vs RMSE ({dataset})', fontsize=16)
        plt.xlabel('Avg_Correlation', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        output_fig_path = os.path.join(output_path, "fig")
        if not os.path.exists(output_fig_path):
            os.makedirs(output_fig_path)
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, f"Scatter Plot of Avg_Correlation vs RMSE ({dataset}).png"))
        #plt.show()

        spearman_corr, p_value = spearmanr(merged_data_imputed['Avg_Correlation'], merged_data_imputed['RMSE'])
        f.write(f"Spearman 相关系数（数据集：{dataset}）：\n")
        f.write(f"相关系数: {spearman_corr:.4f}\n")
        f.write(f"P 值: {p_value:.4f}\n")
        f.write("\n" + "-" * 70 + "\n")