import pandas as pd

df = pd.read_csv('test.csv')
x = df[['Kilo', 'Boy', 'Yas']]
print(x)
