import pandas as pd

df = pd.read_csv("test_jobs.csv", encoding='unicode_escape')
print(df, end='\n\n')

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False, dtype='int8') #smaller dtype would be better

ohe.fit(df[['Renk Tercihi', 'Meslek']])
ohe_data = ohe.transform(df[['Renk Tercihi', 'Meslek']])
print(ohe_data, end='\n\n')

df.drop('Renk Tercihi', axis=1, inplace=True) 
df.drop('Meslek', axis=1, inplace=True)

df[ohe.categories_[0]] = ohe_data[:, :3] 
df[ohe.categories_[1]] = ohe_data[:, 3:]
print(df, end='\n\n')
