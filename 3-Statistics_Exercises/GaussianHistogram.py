import statistics

nd = statistics.NormalDist()
result = nd.samples(10_000)

import matplotlib.pyplot as plt

plt.hist(result, bins=20)
plt.show()
