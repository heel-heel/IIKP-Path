import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def process_datasets():
    base_path = "../Datasets"
    datasets = {
        # "M4-Hourly": {"target_column": "V2", "nonnumerical_column": "V1"},
        "M4-Daily": {"target_column": "V2", "nonnumerical_column": "V1"},
        "M4-Weekly": {"target_column": "V2", "nonnumerical_column": "V1"},
        "M4-Monthly": {"target_column": "V2", "nonnumerical_column": "V1"},
        "M4-Quarterly": {"target_column": "V2", "nonnumerical_column": "V1"},
        "M4-Yearly": {"target_column": "V2", "nonnumerical_column": "V1"},

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

            '''
            # 绘制奇异值衰减图
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(s) + 1), s, 'bo-', linewidth=2, markersize=6)
            plt.xlabel('奇异值序号', fontsize=12)
            plt.ylabel('奇异值大小', fontsize=12)
            plt.title(f'数据集 {i + 1} - 奇异值衰减图', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # 使用对数尺度更好地观察衰减
            plt.tight_layout()

            # 保存图像
            output_fig = f'singular values for {dataset}.png'
            plt.savefig(output_fig, dpi=300, bbox_inches='tight')
            plt.close()
            
            '''
            # 计算累计能量占比
            total_energy = np.sum(s ** 2)
            cumulative_energy = np.cumsum(s ** 2)
            cumulative_energy_ratio = cumulative_energy / total_energy

            # 写入txt文件
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
            #print(f"{output_fig} has been saved.")
            print(f"{output_file} has been saved.")
            print(f"  奇异值范围: {s[0]:.4f} - {s[-1]:.4f}")
            print()

        except Exception as e:
            print(f"处理数据集 {dataset} 时出错: {str(e)}")
            continue


if __name__ == "__main__":
    process_datasets()