import pandas as pd
import numpy as np
import os
from statsmodels.imputation import mice

save_path='./null-mice/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def process_and_fill_csv(file_name, output_file_name):
    df = pd.read_csv(file_name)
    df_impute = df.drop(columns=['V1']).copy()
    np.random.seed(0)
    imp = mice.MICEData(df_impute)
    n_imputations = 50
    imputed_data = []

    for _ in range(n_imputations):
        imp.update('V2')
        imputed_data.append(imp.data['V2'].copy())
    df_imputed = pd.concat(imputed_data, axis=1).mean(axis=1)
    df.loc[df['V2'].isna(), 'V2'] = df_imputed
    df.to_csv(os.path.join(save_path,output_file_name), index=False)

#Missing_rate = [10,30,50,70,90]
Missing_rate = [90]
for rate in Missing_rate:
    input_file = f'dirty-{rate}.csv'
    output_file = f'dirty-mice-{rate}.csv'
    process_and_fill_csv(input_file, output_file)