import pandas as pd
import numpy as np
import os
import random


df = pd.read_csv('clean.csv')

save_path='./null-city/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def generate_dirty_csv(percent_missing):
    num_missing = int(len(df) * percent_missing / 100)
    random_indices = random.sample(range(len(df)), num_missing)

    df.loc[random_indices, 'city'] =pd.NA
    filename = f'dirty-city-{percent_missing}.csv'
    df.to_csv(os.path.join(save_path,filename), index=False)
    print(f'File {filename} generated with {percent_missing}% missing values.')

percent_missing_values = [10, 30, 50, 70, 90]
for percent in percent_missing_values:
    generate_dirty_csv(percent)