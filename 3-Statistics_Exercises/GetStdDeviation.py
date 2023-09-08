import numpy as np

def sd(a, ddof=0):
    return np.sqrt(np.sum((a - np.mean(a)) ** 2) / (len(a) - ddof))

a = np.array([1, 2, 3, 4, 5])

result = sd(a)
print(result)
