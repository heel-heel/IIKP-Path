import pandas as pd
from sklearn.impute import KNNImputer


def process_and_fill_csv(file_name, output_file_name):
    df=pd.read_csv(file_name)
    imputer = KNNImputer(n_neighbors=1)  # 你可以根据需要调整n_neighbors的值
    features = df.drop(columns=['V1'])
    imputed_data = imputer.fit_transform(pd.concat([df['V2'], features], axis=1))

    df['V2'] = imputed_data[:, 0]
    df.to_csv(output_file_name, index=False)

Missing_rate = [10, 30, 50, 70, 90]
for rate in Missing_rate:
    input_file = f'dirty-{rate}.csv'
    output_file = f'dirty-hdi-{rate}.csv'
    process_and_fill_csv(input_file, output_file)