import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = np.maximum(x, 0) #Sıfırdan büyükse x'teki değeri verir. Küçükse 0 verir.

plt.title('RELU Function', fontsize=14, pad=20, fontweight='bold')
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)
axis.set_ylim(-10, 10)
axis.set_xlim(-10, 10)
plt.plot(x, y)
plt.show()
