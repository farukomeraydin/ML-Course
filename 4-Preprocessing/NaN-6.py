import pandas as pd

df = pd.read_csv("melb_data.csv")

print(f'Total rows: {df.shape[0]}, Total columns: {df.shape[1]}', end='\n\n')
missing_info = df.isna().sum()
print('Missing values in each column', end='\n\n')
print(missing_info, end='\n\n')

missing_columns = [name for name in df.columns if df[name].isna().any()]
print('Columns with missing values', end='\n\n')
print(missing_columns, end='\n\n')

missing_ratio = df.isna().sum().sum() / df.size #df.size is row x col
print('Ratio of missing values', end='\n\n')
print(missing_ratio, end='\n\n')

total_missing_rows_ratio = df.isna().any(axis=1).sum() / len(df)
print('Rows with missing values ratio', end='\n\n')
print(total_missing_rows_ratio, end='\n\n')

import numpy as np

impute_val = np.round(df['Car'].mean()) #We can use mean value since there are no outliers
df['Car'].fillna(impute_val, inplace=True) #Filling NaN with mean value
#above is equivalent to df['Car'] = df['Car'].fillna(impute_val)

impute_val = np.round(df['BuildingArea'].mean())
df['BuildingArea'].fillna(impute_val, inplace=True)

impute_val = df['YearBuilt'].mode() #Since this column has interval info we used mode 
df['YearBuilt'].fillna(impute_val[0], inplace=True)

impute_val = df['CouncilArea'].mode() 
df['CouncilArea'].fillna(impute_val[0], inplace=True)

print(df.isna().sum().sum()) #should be zero now
