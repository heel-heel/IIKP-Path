import os
import time
import subprocess

Imputation_Algorithms = {
    #'mode': 'null-mode.py',
    #'knn': 'null-knn.py',
    #'hdi': 'null-hdi.py',
    #'mice': 'null-mice.py',
    #'iim': 'null-iim.py',
    #'si': 'null-si.py',
    #'missfi': 'null-missfi.py',
    #'xgbi': 'null-xgbi.py',
    #'gain': 'null-gain.py',
    #'midae': 'null-midae.py'
}
script_base_path = "../Imputation_Algorithms/Categorical"
datasets = {
    "Beers": {"target_column": "city", "unrelated_column": "id"},
    "Flights": {"target_column": "flight", "unrelated_column": "None"},
    "Hospital": {"target_column": "City", "unrelated_column": "ProviderNumber"},
    #"RedWineQuality": {"target_column": "quality", "unrelated_column": "None"},
    #"AvocadoRipeness": {"target_column": "ripeness", "unrelated_column": "None"}
}
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
#Missing_rate = [95]
base_path = "../Datasets"

def run_imputation(input_path, output_path, target_column, unrelated_column, method, rate):
    input_file = os.path.join(input_path, f'dirty-{rate}.csv')
    output_file = os.path.join(output_path, f'dirty-{method}-{rate}.csv')
    script_name = Imputation_Algorithms[method]
    script = os.path.join(script_base_path, script_name)

    start_time = time.time()
    command = [
        'python', script,
        input_file,
        output_file,
        target_column,
        unrelated_column
    ]
    subprocess.run(command, check=True)
    end_time = time.time()

    elapsed_time = end_time - start_time
    with open(time_file, 'a') as f:
        f.write(f"{method} at {rate}% missing rate: {elapsed_time:.4f} seconds\n")
    print(f"Completed 'dirty-{method}-{rate}'.")

if __name__ == "__main__":
    for dataset, columns in datasets.items():
        target_column = columns["target_column"]
        unrelated_column = columns["unrelated_column"]
        input_path = os.path.join(base_path, dataset, "null")
        time_file = os.path.join(script_base_path, f"{dataset}_imputation_time.txt")
        if not os.path.exists(time_file):
            with open(time_file, 'w') as f:
                f.write(f"Imputation Timing Results for {dataset}\n")
                f.write("===================================\n\n")
        for method in Imputation_Algorithms.keys():
            output_path = os.path.join(base_path, dataset, "Imputation", f"null-{method}")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for rate in Missing_rate:
                run_imputation(input_path, output_path, target_column, unrelated_column, method, rate)
    print("All imputation processes completed.")