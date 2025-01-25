import numpy as np

def standard_scaler(dataset):
    result = np.zeros(dataset.shape) #dataset hangi shape'e sahipse o kadar boyutlu zeros oluşcak
    for col in range(dataset.shape[1]):
        result[:, col] = (dataset[:, col] - np.mean(dataset[:, col])) / np.std(dataset[:, col]) 
        
    return result

a = np.array([[12, 20, 9], [2, 5, 6], [4, 8, 9]])

result = standard_scaler(a)
print(a, end='\n\n')
print(result)


def standard_scaler(dataset):
    return (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0) 
        
    return result

a = np.array([[12, 20, 9], [2, 5, 6], [4, 8, 9]])

result = standard_scaler(a)
print(a, end='\n\n')
print(result)
