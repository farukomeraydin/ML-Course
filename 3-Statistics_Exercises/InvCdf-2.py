from scipy.stats import norm

result = norm.ppf([0.50, 0.68, 0.95], 100, 15)
print(result)
