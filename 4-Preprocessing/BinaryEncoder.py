from category_encoders.binary import BinaryEncoder
import pandas as pd

df = pd.read_csv('test_jobs.csv', encoding='unicode_escape')

be = BinaryEncoder()
transformed_data = be.fit_transform(df['Meslek'])
dropped_df = df.drop(['Meslek'], axis=1)

transformed_df = pd.concat([dropped_df, transformed_data], axis=1)
print(transformed_df)
