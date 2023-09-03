import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

plt.subplot(221)
plt.title('Sinüs Grafiği')
plt.xlabel('x ekseni', loc='left', fontsize=14)
plt.ylabel('y ekseni', loc='top', fontsize=14)

axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

x = np.linspace(-6, 6, 100)
y = np.sin(x)

plt.plot(x, y)
plt.plot(-2, 0.5, marker='x', color='red')
x = 2
y = np.sin(x)
plt.plot(x, y, marker='o', color='red')
plt.plot(x, 0, marker='o', color='red')

y = np.linspace(0, y, 30)
x = np.full(30, 2) 
plt.plot(x, y, linestyle='--')
x = np.linspace(0, 2, 100)
y = np.sin(x)
plt.fill_between(x, y)

plt.subplot(222)
plt.title('Kosinüs Grafiği')
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

x = np.linspace(-6, 6, 100)
y = np.cos(x)

plt.plot(x, y)

plt.subplot(223)
plt.title('Tanjant Grafiği')
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

x = np.linspace(-6, 6, 100)
y = np.tan(x)

plt.plot(x, y)

plt.subplot(224)
plt.title('Parabol Grafiği')
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

x = np.linspace(-10, 10, 100)
y = x **2 - 2

plt.plot(x, y)

plt.show()
