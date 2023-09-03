import pandas as pd

df = pd.read_csv('test.csv')
print(df)
print('--------')
print(df.iloc[:, [1,2,3]].mean())
print('--------')
print(df.iloc[:, [1,2,3]].std())
print('--------')
print(df.iloc[:, [1,2,3]].sum())
print('--------')
