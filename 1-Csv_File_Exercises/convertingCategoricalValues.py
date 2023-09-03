import numpy as np

def foo(b):
    return 1 if b == 'Kadin' else 0

d = {4: foo} #4th column will be input of the foo function
a = np.loadtxt('test.csv', delimiter=',', skiprows=1,usecols=[1,2,3,4],
               converters= d, encoding='utf-8')


print(a)
