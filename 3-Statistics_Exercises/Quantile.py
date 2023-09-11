import numpy as np

x = np.random.randint(0, 100, 10)
y = np.sort(x)

print(y)
print(np.median(y))
print(np.quantile(y, 0.25)) #On the left 25 percent will be left.
print(np.quantile(y, 0.75))
