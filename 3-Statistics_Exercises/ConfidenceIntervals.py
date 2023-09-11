import numpy as np
from scipy.stats import norm

sigma = 10 #population std
n = 100 #samples
sample_mean = 65

samples_std = sigma / np.sqrt(n)

lower_bound = norm.ppf(0.025, sample_mean, samples_std) #confidence interval will be 0.95
upper_bound = norm.ppf(0.975, sample_mean, samples_std) 
print(lower_bound, upper_bound)
