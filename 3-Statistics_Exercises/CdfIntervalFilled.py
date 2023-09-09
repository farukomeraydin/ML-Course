import statistics
import numpy as np

nd = statistics.NormalDist(100, 15)

result = nd.cdf(130) - nd.cdf(120)
print(result)

import matplotlib.pyplot as plt

plt.title('P{120 < X <= 130}', pad=20, fontsize=14, fontweight='bold')
x = np.linspace(40, 160, 1000) #mu is 100 so we picked 40-160
y = [nd.pdf(i) for i in x]

axis = plt.gca() #getcurrentaxis
axis.set_ylim(-0.030, 0.030)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_xticks(range(40, 170, 10))

plt.plot(x, y)

x = np.linspace(120, 130, 100) 
y = [nd.pdf(i) for i in x]

axis.fill_between(x, y)
plt.text(120, -0.01, f'P{{120 < X <= 130}} = {result:.3f}', color='blue', fontsize=14)

plt.show()
