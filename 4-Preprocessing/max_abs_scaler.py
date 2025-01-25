import numpy as np

a = np.array([[12, 20, 9], [2, 5, 6], [4, 8, 9]])

from sklearn.preprocessing import MaxAbsScaler

mas = MaxAbsScaler()
mas.fit(a)
result = mas.transform(a)
print(result)
