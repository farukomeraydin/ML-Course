from scipy.stats import shapiro, norm
import matplotlib.pyplot as plt

a = norm.rvs(100, 15, size=1000)
plt.hist(a)
plt.show()

stat, pval = shapiro(a)
print(f'p değeri: {pval}')
