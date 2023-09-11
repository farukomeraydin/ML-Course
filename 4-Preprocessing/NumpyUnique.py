import pandas as pd
import numpy as np

df = pd.read_csv("test_categorical.csv", encoding='unicode_escape')
print(df, end='\n\n')

print(df['Renk Tercihi'].unique())
print(np.unique(df['Renk Tercihi'].to_numpy()))
