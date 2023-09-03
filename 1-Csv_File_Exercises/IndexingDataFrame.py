import pandas as pd

df = pd.read_csv('test.csv')

print(df.loc[[1,2,3], ['Kilo', 'Boy']]) #1-3 rows
print('------------------')
print(df.loc[:, ['Kilo', 'Boy']]) #All rows
print('------------------')
print(df.iloc[[0,2,3], [1,4]]) #Indexing rows and columns
