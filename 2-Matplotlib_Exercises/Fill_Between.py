import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

plt.title('Canny Example')
plt.xlabel('x', loc='left', fontsize=14)
plt.ylabel('y', loc='top', fontsize=14)

axis = plt.gca()
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

x = np.linspace(0, 6, 34)
y1 = np.array([1, 1, 1, 1.5, 1.6, 1.6, 1.7, 1.6, 1.6, 1.6, 1.65, 1.67, 1.7, 2, 2.2, 2.1, 
              2.05, 2.01, 2.02, 2.04, 2.03, 1.80, 1.7, 1.5, 1.2, 1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

plt.plot(x, y1)

y2 = np.full([34], 1.7)
plt.plot(x, y2, marker='o', color='red')

y3 = np.full([34], 0.9)
plt.plot(x, y3, marker='o', color='red')

y4 = np.full([34], 0)

plt.fill_between(x, y1, y2, where=(y1 >= y2), color='green', alpha=0.5)
plt.fill_between(x, y4, y1, where=(x > 4.6), color='red', alpha=0.5)
plt.fill_between(x, y1, y3, where=(y1 >= y3), color='yellow', alpha=0.5)
plt.show()
