import pandas as pd
import matplotlib.pyplot as plt
import os

datasets = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    #"Flights": {"target_column": "flight", "unrelated_column": None},
    #"Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"}
}
select = {"min"}
label_cr = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]

base_path = "../../Datasets"

for dataset, columns in datasets.items():
    input_path = os.path.join(base_path, dataset, "Mechanism", "classification", "discriminability_precision")
    output_path = os.path.join(base_path, dataset, "Mechanism", "classification", "class_balance")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_fig_path = os.path.join(output_path, "fig")
    if not os.path.exists(output_fig_path):
        os.makedirs(output_fig_path)
    for choice in select:
        for rate in label_cr:
            input_file = f'filled-{choice}-{rate}.csv'
            df = pd.read_csv(os.path.join(input_path, input_file))
            target = columns["target_column"]
            target_counts = df[target].value_counts().reset_index()
            target_counts.columns = [f'{target}', 'num']

            output_file = f'class_balance-results-{choice}-{rate}.csv'
            target_counts.to_csv(os.path.join(output_path, output_file), index=False)
            print(f"{output_file} has been saved.")

            plt.figure(figsize=(10, 6))
            plt.pie(target_counts['num'], labels=None, autopct=None, startangle=140)
            plt.title(f'Class Distribution ({choice}-{rate})')
            plt.tight_layout()

            plt.savefig(os.path.join(output_fig_path, f'class_distribution-{choice}-{rate}.png'))
            print(f"class_distribution-{choice}-{rate}.png has been saved.")
            plt.show()

            # 统计不同 count 的个数
            count_distribution = target_counts['num'].value_counts().reset_index()
            count_distribution.columns = ['count_value', 'frequency']

            # 将 count 的分布情况追加到 .txt 文件
            #txt_file_path = os.path.join(output_path, f'class_balance-results-{choice}.txt')
            #with open(txt_file_path, 'a') as f:
            #    f.write(f"\nCount Distribution for {dataset} ({choice}-{rate}):\n")
            #    count_distribution.to_csv(f, index=False, header=True, sep='\t')

            #print(f"Count distribution has been appended to {txt_file_path}.")

            # 绘制 count_value 的折线图
            plt.figure(figsize=(10, 6))
            plt.bar(count_distribution['count_value'], count_distribution['frequency'])
            plt.title(f'Count Value Distribution ({choice}-{rate})')
            plt.xlabel('Count Value')
            plt.ylabel('Frequency')
            plt.tight_layout()

            plt.savefig(os.path.join(output_fig_path, f'count_distribution-{choice}-{rate}.png'))
            print(f"count_value_distribution-{choice}-{rate}.png has been saved.")
            plt.show()

