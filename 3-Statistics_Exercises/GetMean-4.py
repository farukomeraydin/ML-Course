import pandas as pd

df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(df.mean(axis=1)) #means of rows
print(df.mean()) #default means of columns
