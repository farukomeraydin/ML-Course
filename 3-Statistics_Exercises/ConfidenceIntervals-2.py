import numpy as np
from scipy.stats import norm

sigma = 10 #population std
n = 100 #samples
sample_mean = 65

CF = 0.95

samples_std = sigma / np.sqrt(n)

lower_bound = norm.ppf((1- CF) / 2, sample_mean, samples_std) #confidence interval will be 0.95
upper_bound = norm.ppf(1 - (1 - CF) / 2, sample_mean, samples_std) 
print(lower_bound, upper_bound)
