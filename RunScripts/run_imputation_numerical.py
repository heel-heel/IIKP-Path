import os
import subprocess

Imputation_Algorithms = {
    #'mean': 'null-mean.py',
    #'median': 'null-median.py',
    #'knn': 'null-knn.py',
    #'hdi': 'null-hdi.py',
    #'mice': 'null-mice.py',
    #'iim': 'null-iim.py',
    #'si': 'null-si.py',
    'mfi': 'null-mfi.py',
    'rf': 'null-rf.py',
    'xgbi': 'null-xgbi.py',
    'gain': 'null-gain.py',
    'midae': 'null-midae.py'
}
script_base_path = "../Imputation_Algorithms/Numerical"
datasets = {
    "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
    "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"}
}
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
base_path = "../Datasets"

def run_imputation(input_path, output_path, target_column, nonnumerical_column, method, rate):
    input_file = os.path.join(input_path, f'dirty-{rate}.csv')
    output_file = os.path.join(output_path, f'dirty-{method}-{rate}.csv')
    script_name = Imputation_Algorithms[method]
    script = os.path.join(script_base_path, script_name)
    command = [
        'python', script,
        input_file,
        output_file,
        target_column,
        nonnumerical_column
    ]
    subprocess.run(command, check=True)
    print(f"Completed 'dirty-{method}-{rate}'.")

if __name__ == "__main__":
    for dataset, columns in datasets.items():
        target_column = columns["target_column"]
        nonnumerical_column = columns["nonnumerical_column"]
        input_path = os.path.join(base_path, dataset, "null")
        for method in Imputation_Algorithms.keys():
            output_path = os.path.join(base_path, dataset, "Imputation", f"null-{method}")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for rate in Missing_rate:
                run_imputation(input_path, output_path, target_column, nonnumerical_column, method, rate)
    print("All imputation processes completed.")