import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title('Sinüs Grafiği')

x = np.linspace(-6, 6, 100)
y = np.sin(x)

plt.plot(x, y)

plt.subplot(2, 2, 2)
plt.title('Kosinüs Grafiği')

x = np.linspace(-6, 6, 100)
y = np.cos(x)

plt.plot(x, y)

plt.subplot(2, 2, 3)
plt.title('Tanjant Grafiği')

x = np.linspace(-6, 6, 100)
y = np.tan(x)

plt.plot(x, y)

plt.subplot(2, 2, 4)
plt.title('Parabol Grafiği')

x = np.linspace(-10, 10, 100)
y = x **2 - 2

plt.plot(x, y)

plt.show()
