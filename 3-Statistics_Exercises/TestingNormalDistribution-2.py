from scipy.stats import norm, kstest
import matplotlib.pyplot as plt

a = norm.rvs(size=1000)

plt.hist(a)
plt.show()

stat, pval = kstest(a, 'norm')
print(f'p değeri: {pval}')
