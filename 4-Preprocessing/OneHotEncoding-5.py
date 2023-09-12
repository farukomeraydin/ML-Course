import pandas as pd

df = pd.read_csv("test_jobs.csv", encoding='unicode_escape')
print(df, end='\n\n')

import numpy as np

color_cats = np.unique(df['Renk Tercihi'].to_numpy())
occupation_cats = np.unique(df['Meslek'].to_numpy())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Renk Tercihi'] = le.fit_transform(df['Renk Tercihi'])
df['Meslek'] = le.fit_transform(df['Meslek'])
print(df, end='\n\n')

from tensorflow.keras.utils import to_categorical

ohe_color = to_categorical(df['Renk Tercihi'])
ohe_occupation = to_categorical(df['Meslek'])

df[color_cats] = ohe_color
df[occupation_cats] = ohe_occupation

df.drop(['Renk Tercihi', 'Meslek'], axis=1, inplace=True)
print(df, end='\n\n')
