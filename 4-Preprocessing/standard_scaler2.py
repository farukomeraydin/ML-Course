import numpy as np
import pandas as pd

df = pd.read_csv('diabetes.csv')

dataset_x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
dataset_y = df.iloc[:, -1].to_numpy(dtype=np.float32)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset_x)

transformed_dataset_x = ss.transform(dataset_x)
