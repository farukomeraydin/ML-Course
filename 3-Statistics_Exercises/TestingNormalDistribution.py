from scipy.stats import uniform, kstest
import matplotlib.pyplot as plt

x = uniform.rvs(size=1000)

stat, p = kstest(x, 'norm')
print(stat, p)

plt.hist(x)
plt.show()
