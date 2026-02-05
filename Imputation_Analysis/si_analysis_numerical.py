import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def process_datasets():
    base_path = "../Datasets"
    datasets = {
        "concrete": {"target_column": "concrete_compressive_strength", "nonnumerical_column": "None"},
        "CCPP": {"target_column": "PE", "nonnumerical_column": "None"},
        "AirfoilSelfNoise": {"target_column": "SSPL", "nonnumerical_column": "None"},
        "Abalone": {"target_column": "Rings", "nonnumerical_column": "None"},
        "ParisHousing": {"target_column": "price", "nonnumerical_column": "None"},
    }

    for dataset, columns in datasets.items():
        nonnumerical_column = columns["nonnumerical_column"]
        try:
            input_file = os.path.join(base_path, dataset, "clean.csv")
            df = pd.read_csv(input_file)
            output_path = os.path.join("Results_numerical", "si_analysis")
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            if nonnumerical_column != "None":
                X = df.drop(nonnumerical_column, axis=1).values
            else:
                X = df.values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)

            # calculate
            total_energy = np.sum(s ** 2)
            cumulative_energy = np.cumsum(s ** 2)
            cumulative_energy_ratio = cumulative_energy / total_energy

            output_file = os.path.join(output_path, "si_analysis_results.txt")
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n\n数据集: {dataset}\n")
                f.write(f"数据形状: {X_scaled.shape}\n")
                f.write(f"奇异值数量: {len(s)}\n")
                f.write(f"总能量: {total_energy:.6f}\n")

                f.write("奇异值分析结果:\n")
                f.write("-" * 60 + "\n")
                f.write("序号\t奇异值\t\t累计能量占比\t解释方差占比\n")
                f.write("-" * 60 + "\n")

                for j, (singular_val, energy_ratio) in enumerate(zip(s, cumulative_energy_ratio)):
                    explained_variance = (singular_val ** 2) / total_energy
                    f.write(f"{j + 1}\t{singular_val:.6f}\t{energy_ratio:.6f}\t\t{explained_variance:.6f}\n")

                f.write("\n重要统计信息:\n")
                f.write(f"最大奇异值: {s[0]:.6f}\n")
                f.write(f"最小奇异值: {s[-1]:.6f}\n")
                f.write(f"奇异值中位数: {np.median(s):.6f}\n")
                f.write(f"奇异值平均值: {np.mean(s):.6f}\n")


            print(f"{dataset} is completed.")
            print(f"{output_file} has been saved.")
            print(f"  奇异值范围: {s[0]:.4f} - {s[-1]:.4f}")
            print()

        except Exception as e:
            print(f"Error when processing {dataset}: {str(e)}")
            continue


if __name__ == "__main__":
    process_datasets()