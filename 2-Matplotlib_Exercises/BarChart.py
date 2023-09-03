import matplotlib.pyplot as plt

x =  ['Ali', 'Veli', 'Selami', 'Ayşe', 'Fatma']
y = [98, 56, 34, 72, 21]

plt.ylim((0, 150))

plt.bar(x, y, color=['red', 'green', 'yellow', 'magenta', 'black'], width=0.5)

plt.show()
