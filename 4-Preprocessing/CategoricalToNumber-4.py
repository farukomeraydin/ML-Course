import pandas as pd

df = pd.read_csv("test_ordinal.csv", encoding='unicode_escape')

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()
oe.fit(df[['Egitim Durumu']])

transformed_data = oe.transform(df[['Egitim Durumu']])
print(transformed_data)
