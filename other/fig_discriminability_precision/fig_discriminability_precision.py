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

colors = ['blue', 'green', 'red', 'orange', 'purple']
markers = ['o', 's', '^', 'v', '*']
all_data = []

for i, (dataset, columns) in enumerate(datasets.items()):
    input_path = os.path.join("../../Datasets", dataset, "Mechanism", "classification", "discriminability_precision", "results.txt")
    data = pd.read_csv(input_path, sep=",")
    upper_bound = data[data["J_Target"] != 0]
    lower_bound = data[data["J_Target"] == 0]
    all_data.append((dataset, upper_bound["CR_Target"], upper_bound["J_Actual"], lower_bound["CR_Actual"],
                     lower_bound["J_Actual"], colors[i], markers[i]))

plt.figure(figsize=(10, 6))
for dataset, cr_target_upper, j_actual_upper, cr_target_lower, j_actual_lower, color, marker in all_data:
    plt.plot(cr_target_upper, j_actual_upper, label=f"{dataset} upper bound", color=color, marker=marker)
    plt.plot(cr_target_lower, j_actual_lower, label=f"{dataset} lower bound", color=color, marker=marker, linestyle='--')
plt.legend(loc='upper left')
plt.title("Upper bound vs Lower bound")
plt.xlabel("label correctness radio")
plt.ylabel("class discriminability")
plt.tight_layout()
if not os.path.exists("./fig"):
    os.makedirs("./fig")
plt.tight_layout()
plt.savefig(os.path.join("./fig", "Upper bound vs Lower bound.png"))
print("'Upper bound vs Lower bound.png' has been saved")