from scipy.stats import shapiro, uniform
import matplotlib.pyplot as plt

a = uniform.rvs(100, 15, size=100)
plt.hist(a)
plt.show()
stat, pval = shapiro(a)
print(f'p değeri: {pval}')
