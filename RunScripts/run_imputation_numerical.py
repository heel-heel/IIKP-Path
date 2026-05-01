import os
import subprocess
import time

Imputation_Algorithms = {
    #'mean': 'null-mean.py',
    #'median': 'null-median.py',
    #'mode': 'null-mode.py',
    #'knn': 'null-knn.py',
    #'hdi': 'null-hdi.py',
    #'mice': 'null-mice.py',
    #'iim': 'null-iim.py',
    'si': 'null-si.py',
    #'mfi': 'null-mfi.py',
    #'missfi': 'null-missfi.py',
    #'xgbi': 'null-xgbi.py',
    #'gain': 'null-gain.py',
    #'midae': 'null-midae.py'
}
script_base_path = "../Imputation_Algorithms/Numerical"
output_time_path = "../Imputation_Algorithms/Time_Record"
if not os.path.exists(output_time_path):
    os.makedirs(output_time_path)
datasets = {
    #"ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},
    #"ETTm1": {"target_column": "OT", "nonnumerical_column": "date"},
    #"Illness": {"target_column": "OT", "nonnumerical_column": "date"},
    #"Exchange": {"target_column": "OT", "nonnumerical_column": "date"},
    #"Weather": {"target_column": "OT", "nonnumerical_column": "date"},


    #"concrete": {"target_column": "concrete_compressive_strength", "nonnumerical_column": "None"},
    "CCPP": {"target_column": "PE", "nonnumerical_column": "None"},
    #"AirfoilSelfNoise": {"target_column": "SSPL", "nonnumerical_column": "None"},
    #"Abalone": {"target_column": "Rings", "nonnumerical_column": "None"},
    #"ParisHousing": {"target_column": "price", "nonnumerical_column": "None"},
}
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
Mechanism = ["MCAR", "MAR", "MNAR"]
#Mechanism = [ "MAR", "MNAR"]
base_path = "../Datasets"

def run_imputation(input_path, output_path, target_column, nonnumerical_column, method, rate):
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
        nonnumerical_column
    ]
    subprocess.run(command, check=True)
    end_time = time.time()

    elapsed_time = end_time - start_time
    with open(time_file, 'a') as f:
        f.write(f"{method} at {rate}% missing rate: {elapsed_time:.4f} seconds\n")
    print(f"Completed 'dirty-{method}-{rate}'.")

if __name__ == "__main__":
    for dataset, columns in datasets.items():
        for pattern in Mechanism:
            target_column = columns["target_column"]
            nonnumerical_column = columns["nonnumerical_column"]
            input_path = os.path.join(base_path, dataset, "null", f"{pattern}")
            time_file = os.path.join(output_time_path, f"{dataset}_imputation_time_{pattern}.txt")
            if not os.path.exists(time_file):
                with open(time_file, 'w') as f:
                    f.write(f"Imputation Timing Results for {dataset}\n")
                    f.write("===================================\n\n")
            for method in Imputation_Algorithms.keys():
                output_path = os.path.join(base_path, dataset, "Imputation", f"{pattern}", f"null-{method}")
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                for rate in Missing_rate:
                    run_imputation(input_path, output_path, target_column, nonnumerical_column, method, rate)
    print("All imputation processes completed.")