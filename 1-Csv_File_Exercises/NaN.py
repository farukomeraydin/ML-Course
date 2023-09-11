import pandas as pd

df = pd.read_csv("test_nan.csv")

print(df)
print(type(df.iloc[4, 2])) #NaN is a float type
