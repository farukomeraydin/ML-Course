import pandas as pd

df = pd.read_csv("test_categorical.csv", encoding='unicode_escape')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() 
transformed_data = le.fit_transform(df['Cinsiyet'])
print(transformed_data)

transformed_data = le.fit_transform(df['Renk Tercihi'])
print(transformed_data)
