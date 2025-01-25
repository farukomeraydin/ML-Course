import numpy as np
import pandas as pd

df = pd.read_csv('diabetes.csv')

dataset_x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
dataset_y = df.iloc[:, -1].to_numpy(dtype=np.float32)

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)

transformed_training_dataset_x = ss.transform(training_dataset_x)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='Diabetes')
model.add(Dense(64, activation='relu', input_dim=dataset_x.shape[1], name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2')) 
model.add(Dense(1, activation='sigmoid', name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(transformed_training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 210, 10))

plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(15, 5))
plt.title('Epoch-Binary Accuracy Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Binary Accuracy')
plt.xticks(range(0, 210, 10))

plt.plot(hist.epoch, hist.history['binary_accuracy'])
plt.plot(hist.epoch, hist.history['val_binary_accuracy'])
plt.legend(['Binary Accuracy', 'Validation Binary Accuracy'])
plt.show()

transformed_test_dataset_x = ss.transform(test_dataset_x)
eval_result = model.evaluate(transformed_test_dataset_x, test_dataset_y)

for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

predict_data = np.array([[2,148,58,37,128,25.4,0.699,24], [7,114,67,0,0,32.8,0.258,42],
[5,99,79,27,0,29,0.203,32], [0,110,85,30,0,39.5,0.855,39]]) 

transformed_predict_data = ss.transform(predict_data)
predict_result = model.predict(transformed_predict_data)

for i in range(len(predict_result)):
    print(predict_result[i, 0])
    
for i in range(len(predict_result)):
    print('Şeker Hastası' if predict_result[i, 0] > 0.5 else "Şeker Hastası Değil")
