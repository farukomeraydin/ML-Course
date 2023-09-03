import numpy as np

a = np.loadtxt('test.csv', delimiter=',', skiprows=1, dtype=np.object, usecols=[1,2,3,4])

a[:,3] = (a[:,3] == 'Kadin').astype('int')
print(a)
