import pandas as pd

df = pd.read_csv("test_categorical.csv", encoding='unicode_escape')
print(df, end='\n\n')

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

ohe.fit(df[['Renk Tercihi']])
ohe_data = ohe.transform(df[['Renk Tercihi']])
print(ohe_data, end='\n\n')

df.drop('Renk Tercihi', axis=1, inplace=True)
df[ohe.categories_[0]] = ohe_data
print(df, end='\n\n')
