from scipy.stats import norm, kstest
import matplotlib.pyplot as plt

a = norm.rvs(100, 15, size=1000)

plt.hist(a)
plt.show()

stat, pval = kstest(a, cdf='norm', args=(100, 15))
print(f'p değeri: {pval}')
