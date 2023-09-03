import pandas as pd

df = pd.read_csv('test.csv')
print(df)
print('--------------')
bmi = df['Kilo'] / (df['Boy'] / 100)

df.insert(3, 'Body-Mass Index', bmi)
print(df)
