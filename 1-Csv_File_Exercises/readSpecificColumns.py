import numpy as np

a = np.loadtxt('test.csv', delimiter=',', skiprows=1, dtype=np.float32, usecols=[1,2,3])

print(a)
