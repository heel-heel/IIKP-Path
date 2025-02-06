import pandas as pd
import numpy as np
import os
import random

save_path='./null/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in range(1,10):
    df=pd.read_csv('clean.csv')
    random_indices = random.sample(range(len(df)), i*100)
    df.loc[random_indices, 'V2'] = ""
    filename = f'dirty-{i * 10}.csv'
    df.to_csv(os.path.join(save_path,filename), index=False)