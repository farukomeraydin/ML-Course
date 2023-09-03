import pandas as pd

df = pd.read_csv('test.csv')
print(df)

n = df.iloc[:, [1,2,3]].to_numpy()
print(n)
