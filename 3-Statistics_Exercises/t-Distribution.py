from scipy.stats import norm, t
import numpy as np
import matplotlib.pyplot as plt

DF = 20

x = np.linspace(-5, 5, 1000)

fig = plt.gcf()
fig.set_size_inches((10, 8))

axis = plt.gca() 
axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

y_norm = norm.pdf(x)
y_t = t.pdf(x, DF)

plt.plot(x, y_norm, color='blue')
plt.plot(x, y_t, color='red')

plt.legend(['Standard Normal Distribution', 't Distribution'], fontsize=14, loc='upper right')

plt.show()
