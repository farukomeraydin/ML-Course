import pandas as pd

df = pd.read_csv("test_categorical.csv", encoding='unicode_escape')
print(df, end='\n\n')

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

ohe.fit(df[['Renk Tercihi']])
ohe_data = ohe.transform(df[['Renk Tercihi']])
print(ohe_data, end='\n\n')

import numpy as np
categories = np.unique(df['Renk Tercihi'].to_list())

df.drop('Renk Tercihi', axis=1, inplace=True) #delete that column
df[categories] = ohe_data
print(df, end='\n\n')
