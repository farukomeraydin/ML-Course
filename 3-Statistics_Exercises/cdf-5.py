import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

result = norm.cdf(2) - norm.cdf(-2)
print(result)

x = np.linspace(-5, 5, 1000)
y = norm.pdf(x)

axis = plt.gca() 
axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_xticks(range(-4, 5))
axis.text(2, 0.3, f'{result:.3f}', fontsize=14, fontweight='bold')

plt.plot(x, y)

x = np.linspace(-2, 2, 1000)
y = norm.pdf(x)
plt.fill_between(x, y)
axis.arrow(2.5, 0.25, -2, -0.1, width=0.0255)

plt.show()
