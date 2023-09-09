import random

result = [random.gauss(0, 1) for _ in range(10_000)]

import matplotlib.pyplot as plt

plt.hist(result, bins=20)
plt.show()
