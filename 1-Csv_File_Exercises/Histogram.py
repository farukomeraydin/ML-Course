import pandas as pd

df = pd.read_csv('test.csv')
print(df)

df.iloc[:, [1,2,3]].hist()
