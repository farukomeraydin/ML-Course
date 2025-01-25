from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='Sample')
model.add(Dense(32, activation='relu', input_dim=8, name='Hidden-1')) #input_dim'i manuel girdik model olmadığı için ortada
model.add(Dense(32, activation='relu', name='Hidden-2')) 
model.add(Dense(1, activation='sigmoid', name='Output'))

model.load_weights('diabetes-weights.h5')

import numpy as np

predict_data = np.array([[2,141,58,34,128,25.4,0.699,24], [7,114,66,0,0,32.8,0.258,42],
[5,99,74,27,0,29,0.203,32], [0,109,88,30,0,32.5,0.855,38]]) 

predict_result = model.predict(predict_data)

for i in range(len(predict_result)):
    print(predict_result[i, 0])
    
for i in range(len(predict_result)):
    print('Şeker Hastası' if predict_result[i, 0] > 0.5 else "Şeker Hastası Değil")
