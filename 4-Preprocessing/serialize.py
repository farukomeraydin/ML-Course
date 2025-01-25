import pickle

a = [1, 'ali', [2, 3, 4], 'veli']

with open('test.dat', 'wb') as f:
    pickle.dump(a, f)

with open('test.dat', 'rb') as f:
    b = pickle.load(f)

print(b)  
