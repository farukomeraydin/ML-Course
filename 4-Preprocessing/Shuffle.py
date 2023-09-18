import numpy as np

dataset = np.random.randint(0, 100, (10, 11))
dataset_copy = dataset.copy()

np.random.shuffle(dataset_copy) #rows are shuffled

print(dataset)
print(dataset_copy)
