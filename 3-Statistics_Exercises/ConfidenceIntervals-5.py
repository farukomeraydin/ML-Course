import numpy as np
from scipy.stats import norm

population = np.random.randint(0, 1_000_000_000, 1_000_000)
sigma = np.std(population)

CL = 0.95
SAMPLE_SIZE = 100

sample = np.random.choice(population, SAMPLE_SIZE) 
print(f'Population Mean: {np.mean(population)}')

lower_bound, upper_bound = norm.interval(CL, np.mean(sample), sigma / np.sqrt(SAMPLE_SIZE))
print(lower_bound, upper_bound)
