import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.linspace(40, 160, 1000)
y = norm.pdf(x, 100, 15)

plt.plot(x, y)

x = np.full(200, 100)
yend = norm.pdf(100, 100, 15) #Pinnacle point
y = np.linspace(0, yend, 200) #Vertical line
plt.plot(x, y, linestyle='--')

plt.show()
