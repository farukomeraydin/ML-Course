import statistics 
import matplotlib.pyplot as plt 

nd = statistics.NormalDist() 
x = [i * 0.001 for i in range(-5000, 5000)] 
y = [nd.pdf(i) for i in x] 

plt.title('Gauss Eğrisi') 
plt.xlabel('X') 
plt.ylabel('Y') 
plt.plot(x, y) 
plt.show()
