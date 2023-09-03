import statistics 

a = [1, 23, 56, 12, 45, 21]

print(statistics.median(a))

import numpy as np

a = np.random.randint(1, 100, (10, 10))

print(np.median(a, axis=0)) #medians of columns

import pandas as pd

a = np.random.randint(1, 100, (10, 10))
df = pd.DataFrame(a)
print(df.median()) #medians of columns
