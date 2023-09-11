import pandas as pd
import numpy as np

df = pd.read_csv("melb_data.csv")

#np.nan == np.nan gives False. Use pd.isnull or pd.isna(...). Gives the number of NaN in each column
result = pd.isnull(df).sum()
result2 = pd.isnull(df).sum(axis=1) #NaN number of each row
result3 = np.sum(pd.isnull(df).any(axis=1)) #any method gives True if any of them is True.Here we got the number of rows that has NaN.

print(result)
print('------------')
print(result2)
print('------------')
print(result3)
