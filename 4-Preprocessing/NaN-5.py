import pandas as pd

df = pd.read_csv("melb_data.csv")
missing_columns = [name for name in df.columns if df[name].isna().any()] 

print(missing_columns)
print(df[missing_columns])
