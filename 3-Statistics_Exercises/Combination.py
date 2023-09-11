import math

trial = ['T', 'H']
result = [(x, y, z, k) for x in trial for y in trial for z in trial for k in trial]

print(result)
print(math.comb(4, 2)) #2 Heads 
