import pandas as pd

df = pd.read_csv("melb_data.csv")

print(f'Total rows: {df.shape[0]}, Total columns: {df.shape[1]}', end='\n\n')
missing_info = df.isna().sum()
print('Missing values in each column', end='\n\n')
print(missing_info, end='\n\n')

missing_columns = [name for name in df.columns if df[name].isna().any()]
print('Columns with missing values', end='\n\n')
print(missing_columns, end='\n\n')

missing_ratio = df.isna().sum().sum() / df.size 
print('Ratio of missing values', end='\n\n')
print(missing_ratio, end='\n\n')

total_missing_rows_ratio = df.isna().any(axis=1).sum() / len(df)
print('Rows with missing values ratio', end='\n\n')
print(total_missing_rows_ratio, end='\n\n')

import numpy as np
from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean')

"""
si.fit(df[['Car', 'BuildingArea']]) 
df[['Car', 'BuildingArea']] = np.round(si.transform(df[['Car', 'BuildingArea']]))
"""

df[['Car', 'BuildingArea']] = np.round(si.fit_transform(df[['Car', 'BuildingArea']])) ##equivalent to above

si = SimpleImputer(strategy='most_frequent') #In order to not create another object you can do si.set_params(strategy='most_frequent')
"""
si.fit(df[['YearBuilt', 'CouncilArea']]) 
df[['YearBuilt', 'CouncilArea']] = si.transform(df[['YearBuilt', 'CouncilArea']])
"""
df[['YearBuilt', 'CouncilArea']] = si.fit_transform(df[['YearBuilt', 'CouncilArea']]) #equivalent to above
