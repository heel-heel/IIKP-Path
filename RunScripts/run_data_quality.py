import os
import subprocess
import json

datasets = {
    "ETTh1": {"target_column": "OT", "nonnumerical_column": "date"},
    "ETTm1": {"target_column": "OT", "nonnumerical_column": "date"},
    "Illness": {"target_column": "OT", "nonnumerical_column": "date"},
    "Exchange": {"target_column": "OT", "nonnumerical_column": "date"},
    "Weather": {"target_column": "OT", "nonnumerical_column": "date"},

    "concrete": {"target_column": "concrete_compressive_strength", "nonnumerical_column": "None"},
    "CCPP": {"target_column": "PE", "nonnumerical_column": "None"},
    "AirfoilSelfNoise": {"target_column": "SSPL", "nonnumerical_column": "None"},
    "Abalone": {"target_column": "Rings", "nonnumerical_column": "None"},
    "ParisHousing": {"target_column": "price", "nonnumerical_column": "None"},

    "ETTh2": {"target_column": "OT", "nonnumerical_column": "date"},
    "ETTh2-history": {"target_column": "OT", "nonnumerical_column": "date"},
    "ETTh2-test": {"target_column": "OT", "nonnumerical_column": "date"},

    "BostonHousePrice": {"target_column": "MEDV", "nonnumerical_column": "None"},
    "BostonHousePrice-history": {"target_column": "MEDV", "nonnumerical_column": "None"},
    "BostonHousePrice-test": {"target_column": "MEDV", "nonnumerical_column": "None"},
}
Imputation_Algorithms = ['mean', 'median', 'mode', 'knn', 'hdi', 'mice', 'iim', 'si', 'mfi', 'missfi', 'xgbi', 'gain', 'midae']
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

def run_script(script, config_file):
    print(f"Run: {script}")
    command = [
        'python', script,
        config_file
    ]
    subprocess.run(command, check=True)
    print(f"Finish: {script}")

def main():
    base_path = os.path.join("../Data_Quality")
    scripts = [
        "ks_test.py",
        "kl_divergence.py",
        "2_wasserstein_distance.py",
        "sliced_wasserstein_distance.py",
        "mutual_information.py",
    ]
    config = {
        'datasets': datasets,
        'Imputation_Algorithms': Imputation_Algorithms,
        'Missing_rate': Missing_rate
    }
    config_file = 'config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    for script_name in scripts:
        run_script(os.path.join(base_path, script_name), config_file)

if __name__ == "__main__":
    main()