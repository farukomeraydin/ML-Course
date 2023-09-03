import pandas as pd

df = pd.read_csv('test.csv')
print(df)
print('------------')
df.insert(1, 'xxxx', [1,2,3,4,5])
print(df)
