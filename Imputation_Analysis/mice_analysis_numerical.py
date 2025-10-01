import pandas as pd
import numpy as np
import os
from datetime import datetime


def analyze_dataset_correlation(input_file):
    df = pd.read_csv(input_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    return corr_matrix, numeric_cols

def save_correlation_results(datasets, output_file="correlation_results.txt"):
    with open(output_file, 'a', encoding='utf-8') as f:
        for dataset in datasets:
            input_file = os.path.join("../Datasets", dataset, "clean.csv")
            f.write(f"\n\n数据集: {dataset}\n")
            f.write("-" * 60 + "\n")
            corr_matrix, columns = analyze_dataset_correlation(input_file)

            # 创建矩阵表头
            header = "变量名".ljust(19)
            for col in columns:
                # 缩写列名以适应显示
                short_name = col[:8] + ".." if len(col) > 10 else col
                header += f" | {short_name:>10}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for row_idx, row_name in enumerate(columns):
                # 行名显示
                short_row_name = row_name[:18] + ".." if len(row_name) > 20 else row_name
                line = f"{short_row_name:<20}"
                for col_idx, col_name in enumerate(columns):
                    corr_value = corr_matrix.iloc[row_idx, col_idx]
                    if row_idx == col_idx:
                        line += " |   1.000000"
                    else:
                        line += f" | {corr_value:10.6f}"
                f.write(line + "\n")

            f.write("\n")

            # 相关性统计摘要
            f.write("相关性统计摘要:\n")
            # 获取下三角矩阵的值（不包括对角线）
            mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            corr_values = corr_matrix.where(mask).values.flatten()
            corr_values = corr_values[~np.isnan(corr_values)]

            if len(corr_values) > 0:
                f.write(f"  相关系数平均值: {np.mean(corr_values):.3f}\n")
                f.write(f"  相关系数标准差: {np.std(corr_values):.3f}\n")
                f.write(f"  最大相关系数: {np.max(corr_values):.3f}\n")
                f.write(f"  最小相关系数: {np.min(corr_values):.3f}\n")
            f.write("=" * 70 + "\n\n")


def main():
    datasets = [
        #"M4-Hourly",

        #'M4-Daily',
        #'M4-Weekly',
        #'M4-Monthly',
        #'M4-Quarterly',
        #'M4-Yearly',
        #"concrete",
        #"CCPP",
        #"AirfoilSelfNoise",
        "Abalone",
        #"ParisHousing"
    ]
    output_path = os.path.join("Results_numerical", "mice_analysis")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, "mice_analysis_results.txt")
    save_correlation_results(datasets, output_file)
    print(f"{output_file} has been saved.")


# 如果直接运行此脚本
if __name__ == "__main__":
    main()