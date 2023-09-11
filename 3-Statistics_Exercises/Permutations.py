import itertools

for t in itertools.permutations([1, 2, 3], 2): #Binary permutation. n-permutation = n!/(n-k)
    print(t)
