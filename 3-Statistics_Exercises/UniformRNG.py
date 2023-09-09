from scipy.stats import uniform
import numpy as np

x = np.random.random(10)
print(x)

x = uniform.rvs(0, 1, 10) #same as above
print(x)
