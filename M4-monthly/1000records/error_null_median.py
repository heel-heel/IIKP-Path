import pandas as pd
import os

save_path='./null-median/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

read_path = './null/'

def process_and_fill_csv(file_name, output_file_name):
    data = pd.read_csv(os.path.join(read_path,file_name))
    global_median = data['V2'].median()
    data['V2'] = data['V2'].fillna(global_median)
    data.to_csv(os.path.join(save_path,output_file_name), index=False)

Missing_rate = [10, 30, 50, 70, 90]
for rate in Missing_rate:
    input_file = f'dirty-{rate}.csv'
    output_file = f'dirty-median-{rate}.csv'
    process_and_fill_csv(input_file, output_file)