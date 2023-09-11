import pandas as pd

df = pd.read_csv("melb_data.csv")

df_deleted = df.dropna(axis=0) #rows eliminated
print(df_deleted)
