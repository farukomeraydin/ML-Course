import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10), facecolor='#FFFFDD')
plt.title('Sinüs Eğrisi', fontsize=14, fontweight='bold', color='red', pad=20)
plt.xlim((-20, 20))
plt.xticks(range(-20, 21))
plt.yticks(np.arange(-1, 1, 0.1))


x = np.linspace(-10, 10, 100) #[-10, 10] aralığında 100 nokta verir
y = np.sin(x)
z = np.cos(x)

plt.plot(x, y, color='red', linewidth=4, marker='o', markersize=10,markerfacecolor='green')

plt.text(-18, 0.7, "This is a test", fontsize=14, color='red')
plt.arrow(-17, 0, 1, 1, linewidth=4)

plt.plot(x, z, color='blue', linewidth=4)

plt.legend(['Sinüs', 'Kosinüs'], loc='upper right', fontsize=14, facecolor='yellow')
plt.grid()
plt.show()
