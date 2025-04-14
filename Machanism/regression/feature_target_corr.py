import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

datasets = {
    "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"}
}
Imputation_Algorithms = ['mean', 'median', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'rf', 'xgbi', 'gain', 'midae']
Missing_rate = [10, 30, 50, 70, 90]

for dataset, columns in datasets.items():
    target_column = columns["target_column"]
    nonnumerical_column = columns["nonnumerical_column"]
    base_path = "../../Datasets"
    output_path = os.path.join(base_path, dataset, "Machanism", "regression", "feature_target_corr")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results = pd.DataFrame(columns=['file_name', 'Max_Correlation', 'Min_Correlation', 'Avg_Correlation'])
    for model in Imputation_Algorithms:
        for rate in Missing_rate:
            input_dirty_file = f'dirty-{model}-{rate}.csv'
            try:
                input_dirty_path = os.path.join(base_path, dataset, "Imputation", f"null-{model}")
                dirty_df = pd.read_csv(os.path.join(input_dirty_path, input_dirty_file))
                df_numeric = dirty_df.drop(columns=[nonnumerical_column], errors='ignore')

                correlation_matrix = df_numeric.corr()
                if target_column in correlation_matrix.columns:
                    target_correlations = correlation_matrix[target_column].drop(target_column, errors='ignore')

                    max_corr = target_correlations.max()
                    min_corr = target_correlations.min()
                    avg_corr = target_correlations.mean()

                    new_row = pd.DataFrame({
                        'file': [input_dirty_file],
                        'Max_Correlation': [max_corr],
                        'Min_Correlation': [min_corr],
                        'Avg_Correlation': [avg_corr]
                    })

                    results = pd.concat([results, new_row], ignore_index=True)
            except Exception as e:
                print(f"Error processing file {input_dirty_file}: {e}")

    results.to_csv(os.path.join(output_path, 'feature_target_corr_results.csv'), index=False)
    print(f"相关性结果已保存到 {os.path.join(output_path, 'feature_target_corr_results.csv')}")

    results['model'] = results['file_name'].apply(lambda x: x.split('-')[1])
    results['percentage'] = results['file_name'].apply(lambda x: int(x.split('-')[2].replace('.csv', '')))
    heatmap_data = results.pivot(index='percentage', columns='model', values='Avg_Correlation')
    heatmap_data = heatmap_data.reindex(columns=Imputation_Algorithms)

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='Blues', fmt=".4f", linewidths=.5)
    plt.title(f'Average Feature-Target Correlation Heatmap')
    plt.xlabel('Model')
    plt.ylabel('Missing rate')
    plt.xticks(rotation=45, ha='right')

    output_fig_path = os.path.join(output_path, "fig")
    if not os.path.exists(output_fig_path):
        os.makedirs(output_fig_path)
    plt.savefig(os.path.join(output_fig_path, f'Average_Feature_Target_Corr_Heatmap.png'))
    plt.show()