import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

datasets = {
    "concrete": {"target_column": "concrete_compressive_strength"},
    #"CCPP": {"target_column": "PE", "nonnumerical_column": "None"},
    #"AirfoilSelfNoise": {"target_column": "SSPL", "nonnumerical_column": "None"}
}
def normalize_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    numeric_cols = df.columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    df.to_csv(output_file, index=False)
    print(f"归一化完成！结果已保存至: {output_file}")

for dataset, columns in datasets.items():
    input_file = os.path.join("../Datasets", dataset, "origin_clean.csv")
    output_file = os.path.join("../Datasets", dataset, "clean.csv")
    normalize_csv(input_file, output_file)