import pandas as pd

df = pd.read_csv("test_ordinal.csv", encoding='unicode_escape', converters={6: lambda s: {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'g': 4}[s]}) #encoding in 6th column
print(df)
