import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

save_path='./null-xgboost/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def process_and_fill_csv(file_name, output_file_name):
    data = pd.read_csv(file_name)
    model = XGBRegressor()
    imputer = IterativeImputer(estimator=model, max_iter=30, random_state=0)
    features = data.columns.drop('V1')
    data_imputed = imputer.fit_transform(data[features])
    data_imputed = pd.DataFrame(data_imputed, columns=features)
    data['V2'] = data_imputed['V2']
    data.to_csv(os.path.join(save_path,output_file_name), index=False)

Missing_rate = [10, 30, 50, 70, 90]
for rate in Missing_rate:
    input_file = f'dirty-{rate}.csv'
    output_file = f'dirty-xgboost-{rate}.csv'
    process_and_fill_csv(input_file, output_file)