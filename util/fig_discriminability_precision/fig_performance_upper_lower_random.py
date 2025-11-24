import matplotlib.pyplot as plt
import pandas as pd
import os

datasets = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    "Flights": {"target_column": "flight", "unrelated_column": None},
    "Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"},
    "RedWineQuality": {"target_column": "quality", "unrelated_column": "None"},
    "AvocadoRipeness": {"target_column": "ripeness", "unrelated_column": "None"}
}
label_cr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

colors = ['blue', 'green', 'red', 'orange', 'purple']
markers = ['o', 's', '^', 'v', '*']

for i, (dataset, info) in enumerate(datasets.items()):
    plt.figure(figsize=(10, 6))
    max_min_input_path = os.path.join("../../Downstream_Results", "classification", dataset, f"mlp-discriminability_precision-results-{dataset}.csv")
    max_min_data = pd.read_csv(max_min_input_path)

    random_input_path = os.path.join("../../Downstream_Results", "classification", dataset, f"mlp-discriminability_precision_random-results-{dataset}.csv")
    random_data = pd.read_csv(random_input_path)

    max_data = max_min_data[max_min_data['File Name'].str.contains('filled-max-')]
    min_data = max_min_data[max_min_data['File Name'].str.contains('filled-min-')]
    max_values = max_data['PG(F1 Score)'].values
    min_values = min_data['PG(F1 Score)'].values

    plt.plot(label_cr, max_values, label=f"max", color=colors[0], marker=markers[0])
    plt.plot(label_cr, min_values, label=f"min", color=colors[1], marker=markers[1], linestyle='--')

    for j in range(1, 6):
        random_values = []
        for cr in label_cr:
            file_name = f"random-{cr}-{j}.csv"
            row = random_data[random_data['File Name'] == file_name]
            if not row.empty:
                random_values.append(row['PG(F1 Score)'].values[0])
            else:
                random_values.append(None)
        plt.plot(label_cr, random_values, label=f"random-{j}", color=colors[2], marker=markers[2], linestyle=':')

    plt.legend(loc='upper right')
    plt.title(f"Upper bound vs Lower bound vs Random for {dataset}")
    plt.xlabel("label correctness radio")
    plt.ylabel("PG(F1 Score)")
    plt.tight_layout()
    if not os.path.exists("./fig"):
        os.makedirs("./fig")
    plt.tight_layout()
    plt.savefig(os.path.join("./fig", f"Upper bound vs Lower bound vs Random for {dataset}"))
    print(f"'Upper bound vs Lower bound vs Random for {dataset}' has been saved.")