import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100) #100 points between [-10, 10]
y = np.sin(x)

plt.plot(x, y)
