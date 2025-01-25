import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = (np.e ** (2 * x) - 1) / (np.e ** (2 * x) + 1)

plt.title('Hyperbolic Tangent (tanh) Function', fontsize=14, pad=20, fontweight='bold')
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_ylim(-1, 1)
axis.set_xlim(-10, 10)

plt.plot(x, y)
plt.show()
