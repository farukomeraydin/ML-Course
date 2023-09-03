import pandas as pd

df = pd.read_csv('test.csv')
print(df)

x = df.drop(['Kilo'], axis=1)
print(x)
