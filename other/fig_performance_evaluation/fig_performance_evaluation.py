import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置参数
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
Imputation_Algorithms = ['mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'missfi', 'xgbi', 'gain', 'midae']

# 颜色和线型设置
colors = plt.cm.tab20(np.linspace(0, 1, len(Imputation_Algorithms)))
line_styles = ['-', '--', '-.', ':'] * 4
markers = ['o', 'v', '+', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']

# 任务和对应的数据集及模型
tasks = {
    'timeseries': {
        'datasets': ['M4-Monthly', 'M4-Quarterly', 'M4-Yearly'],
        'models': ['mlp', 'tsmixer']
    },
    #'classification': {
    #    'datasets': ['Beers', 'Flights', 'Hospital'],
    #    'models': ['mlp']
    #},
    #'regression': {
    #    'datasets': ['M4-Monthly', 'M4-Quarterly', 'M4-Yearly'],
    #    'models': ['mlp']
    #}
}

# 每个任务对应的纵坐标和标题
task_metrics = {
    #'classification': {'metric': 'PG(F1 Score)', 'title': 'Classification:PG vs Missing Rate'},
    'timeseries': {'metric': 'PG(RMSE)', 'title': 'Time Series Forecasting:PG vs Missing Rate'},
    #'regression': {'metric': 'PG(MAE)', 'title': 'Regression:PG vs Missing Rate'}
}

output_dir = "./fig/"
os.makedirs(output_dir, exist_ok=True)

for task, task_info in tasks.items():
    datasets = task_info['datasets']
    models = task_info['models']

    for dataset in datasets:
        for model in models:
            input_file = os.path.join("../../Downstream_Results", task, dataset,
                                      f"{model}-imputation-results-{dataset}.csv")

            if not os.path.exists(input_file):
                print(f"文件 {input_file} 不存在，跳过该数据集。")
                continue

            df = pd.read_csv(input_file)
            dirty_rates = []
            dirty_pg_values = []
            for rate in Missing_rate:
                row = df[df['File Name'] == f"dirty-{rate}.csv"]
                if not row.empty:
                    dirty_rates.append(rate)
                    dirty_pg_values.append(row[task_metrics[task]['metric']].values[0])

            plt.figure(figsize=(14, 8))
            plt.plot(dirty_rates, dirty_pg_values,
                     label='Dirty (No Imputation)',
                     color='black',
                     linewidth=3,
                     linestyle='-',
                     marker='o',
                     markersize=8)

            for i, imp_model in enumerate(Imputation_Algorithms):
                model_rates = []
                model_pg_values = []

                for rate in Missing_rate:
                    row = df[df['File Name'] == f"dirty-{imp_model}-{rate}.csv"]
                    if not row.empty:
                        model_rates.append(rate)
                        model_pg_values.append(row[task_metrics[task]['metric']].values[0])

                if model_rates:
                    plt.plot(model_rates, model_pg_values,
                             label=imp_model.upper(),
                             color=colors[i],
                             linestyle=line_styles[i],
                             linewidth=2,
                             marker=markers[i],
                             markersize=6)

            plt.title(f"{task_metrics[task]['title']} ({dataset}, Model: {model.upper()})", fontsize=16, pad=20)
            plt.xlabel('Missing Rate (%)', fontsize=14)
            plt.ylabel("PG", fontsize=14)
            plt.xticks(Missing_rate, rotation=45, fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='upper left',
                       fontsize=10,
                       framealpha=1,
                       edgecolor='black')

            plt.tight_layout()
            task_output_dir = os.path.join(output_dir, task, model)
            os.makedirs(task_output_dir, exist_ok=True)
            plt.savefig(os.path.join(task_output_dir, f'{model}_Performance_Evaluation_{dataset}.png'),
                        dpi=300,
                        bbox_inches='tight',
                        transparent=False)
            plt.close()