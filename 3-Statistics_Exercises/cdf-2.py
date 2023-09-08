import statistics

nd = statistics.NormalDist(100, 15)
result = nd.cdf(140) - nd.cdf(130) #P(130<x<140)
print(result)
