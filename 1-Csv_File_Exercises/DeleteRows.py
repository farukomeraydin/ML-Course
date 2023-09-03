import pandas as pd

df = pd.read_csv('test.csv')
print(df)
print('------------')
y = df.drop([1,3,4], axis=0)
print(y)
