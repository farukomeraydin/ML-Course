import pandas as pd

df = pd.read_csv('test.csv')
print(df)
print('------------')
concat_df = pd.concat([df, pd.Series([1,2,3,4,5], name='xxx')], axis=1)
print(concat_df)
