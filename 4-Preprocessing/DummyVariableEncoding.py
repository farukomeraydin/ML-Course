import pandas as pd

df = pd.read_csv('test_colors.csv', encoding='unicode_escape')
print(df, end='\n\n')

ohe_df = pd.get_dummies(df, columns=['Renk Tercihi'], drop_first=True) #For dummy variable encoding drop_first=True

print(ohe_df)
