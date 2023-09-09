from scipy.stats import uniform

result = uniform.cdf(0.5)
print(result)

result = uniform.cdf(0.8)
print(result)

result = uniform.pdf([0.1, 0.2, 0.3])
print(result)

result = uniform.ppf([0.1, 0.2, 0.3])
print(result)
