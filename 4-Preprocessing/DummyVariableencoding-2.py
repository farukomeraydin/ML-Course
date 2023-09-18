import pandas as pd

df = pd.read_csv('test_colors.csv', encoding='unicode_escape')
print(df, end='\n\n')

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, drop='first')

transformed_data = ohe.fit_transform(df[['Renk Tercihi']])

print(transformed_data)
