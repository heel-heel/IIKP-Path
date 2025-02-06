import pandas as pd
import numpy as np
import random

df=pd.read_csv('clean.csv')
random_indices = random.sample(range(len(df)), 100)
df.loc[random_indices, 'Ozone'] = ""

df.to_csv('dirty-10.csv', index=False)