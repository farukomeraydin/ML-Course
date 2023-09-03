import pandas as pd

df = pd.read_csv('test.csv')
print(df)
print('------------')
df.drop(['Adi Soyadi'], axis=1, inplace=True)
print(df)
