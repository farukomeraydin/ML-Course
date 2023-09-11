import pandas as pd

df = pd.read_csv("test_categorical.csv", encoding='unicode_escape')

from sklearn.preprocessing import LabelEncoder
import numpy as np

le = LabelEncoder()
le.fit(df['Renk Tercihi'])

result = le.inverse_transform(np.array([2, 1, 1, 1, 2, 2, 1, 0]))
print(result)
