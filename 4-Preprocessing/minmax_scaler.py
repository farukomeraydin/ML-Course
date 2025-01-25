import numpy as np

def minmax_scaler(dataset):
    return (dataset - np.min(dataset, axis=0)) / (np.max(dataset, axis=0) - np.min(dataset, axis=0))

a = np.array([[12, 20, 9], [2, 5, 6], [4, 8, 9]])

result = minmax_scaler(a)
print(a, end='\n\n')
print(result)

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
mms.fit(a)
result = mms.transform(a)
print(result)
