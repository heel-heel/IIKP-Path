import pandas as pd
import numpy as np
import random

for i in range(1,10):
    df=pd.read_csv('clean.csv')
    random_indices = random.sample(range(len(df)), i*500)
    df.loc[random_indices, 'V2'] = ""
    filename = f'dirty-{i * 10}.csv'
    df.to_csv(filename, index=False)