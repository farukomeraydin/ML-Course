import numpy as np

a = np.array([[12, 20, 9], [2, 5, 6], [4, 8, 9]])


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(a)

result = ss.transform(a)
print(result)
