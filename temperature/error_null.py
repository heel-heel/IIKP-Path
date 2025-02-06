import pandas as pd
import numpy as np
import random

df=pd.read_csv('clean.csv')
random_indices = random.sample(range(len(df)), 3210)
df.loc[random_indices, 'Temp'] = ""

df.to_csv('dirty-90.csv', index=False)