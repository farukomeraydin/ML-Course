import pandas as pd

df = pd.read_csv("test_jobs.csv", encoding= 'unicode_escape')
print(df, end='\n\n')

import numpy as np

color_cats = np.unique(df['Renk Tercihi'].to_numpy())
occupation_cats = np.unique(df['Meslek'].to_numpy())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Renk Tercihi'] = le.fit_transform(df['Renk Tercihi'])
df['Meslek'] = le.fit_transform(df['Meslek'])
print(df, end='\n\n')

color_um = np.eye(len(color_cats))
occupation_um = np.eye(len(occupation_cats))

ohe_color = color_um[df['Renk Tercihi'].to_numpy()]
ohe_occupation = occupation_um[df['Meslek'].to_numpy()]

df.drop(['Renk Tercihi', 'Meslek'], axis=1, inplace=True)

df[color_cats] = ohe_color
df[occupation_cats] = ohe_occupation
print(df, end='\n\n')
