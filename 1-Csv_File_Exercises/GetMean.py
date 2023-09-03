import pandas as pd

df = pd.read_csv('test.csv')
print(df)

print(df.iloc[:, [1,2,3]].mean())
