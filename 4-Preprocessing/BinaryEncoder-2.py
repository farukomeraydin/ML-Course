from category_encoders.binary import BinaryEncoder
import pandas as pd

df = pd.read_csv('test_jobs.csv', encoding='unicode_escape')

be = BinaryEncoder(cols=['Meslek'])

transformed_df = be.fit_transform(df)

print(transformed_df)
