import pandas as pd

df = pd.read_csv("test_categorical.csv", encoding='unicode_escape')

from sklearn.preprocessing import LabelEncoder

print(df, end='\n\n')
le = LabelEncoder()

df['Renk Tercihi'] = le.fit_transform(df['Renk Tercihi'])
df['Cinsiyet'] = le.fit_transform(df['Cinsiyet'])

print(df)
