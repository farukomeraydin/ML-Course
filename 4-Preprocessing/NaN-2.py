import pandas as pd
import numpy as np

df = pd.read_csv("test_nan.csv")
print(df)

df[df == 'None'] = np.nan 
print(df)
