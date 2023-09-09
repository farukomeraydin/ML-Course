import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(0, 1_000_000_000, 1_000_000)
samples = np.random.choice(x, (1_000_000, 50)) #10k times 50 batch samples are randomly chosen
samples_means = np.mean(samples, axis=1) #mean of each row

plt.hist(samples_means, bins=50) #sample means have normal distributions according to central limit theorem

population_mean = np.mean(x)
sample_means_mean = np.mean(samples_means)
population_std = np.std(x) 
sample_means_std = np.std(samples_means)

plt.show()

print(f'Anakütle ortalaması: {population_mean}')
print(f'Örnek ortalamalarının ortalaması: {sample_means_mean}')
print(f'Fark: {np.abs(population_mean - sample_means_mean)}')

print(f'Anakütle standart sapması / sqrt(50) = {population_std / np.sqrt(50)}')
print(f'Örnek ortalamalarının standart sapması: {sample_means_std}')

import math

print(math.comb(1_000_000_000, 50)) #this many samples will have same mean as entire population
