import pandas as pd

df = pd.read_csv("test_jobs.csv", encoding='unicode_escape')
print(df, end='\n\n')

ohe_df = pd.get_dummies(df, columns=['Renk Tercihi', 'Meslek'], prefix=['', ''], prefix_sep='') #We can rename columns and add prefixes
print(ohe_df)
