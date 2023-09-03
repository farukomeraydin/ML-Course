import numpy as np


d = {4: lambda s: 1 if s == 'Kadin' else 0} #4th column will be input of the lambda expression
a = np.loadtxt('test.csv', delimiter=',', skiprows=1,usecols=[1,2,3,4],
               converters= d, encoding='utf-8')


print(a)
