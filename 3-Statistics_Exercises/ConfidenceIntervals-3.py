import numpy as np
from scipy.stats import norm

sigma = 10 
n = 50 
sample_mean = 65 

CF = 0.99

for n in range(30, 100, 5):
    samples_std = sigma / np.sqrt(n)
    
    lower_bound = norm.ppf((1- CF) / 2, sample_mean, samples_std) 
    upper_bound = norm.ppf(1 - (1 - CF) / 2, sample_mean, samples_std) 
    print(f'{n}: [{lower_bound}, {upper_bound}]')
