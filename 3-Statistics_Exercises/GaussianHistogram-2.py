from scipy.stats import norm
import matplotlib.pyplot as plt

x = norm.rvs(100, 15, 100_000)
plt.hist(x, bins=20)
plt.show()
