import pandas as pd

df = pd.read_csv('auto-mpg.data', delimiter=r'\s+', header=None) #csv dosyası , ister. Ancak bu veri kümesi boşluklarla ayrılmış.Regular expression girmek gerek.Birden fazla boşluk da var çünkü.

df = df.iloc[:, :-1]
dataset_df = df[df.iloc[:, 3] != '?'] #Eksik verileri attık.

dataset_ohe_df = pd.get_dummies(dataset_df, columns=[7]) #7.sütunu ohe yaptık

dataset = dataset_ohe_df.to_numpy(dtype='float32')

dataset_x = dataset[:, 1:] #1 tane feature'u kullanmayalım dedik
dataset_y = dataset[:, 0] #Yakıt tüketimini tahmin edeceğiz

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
mms.fit(training_dataset_x)

scaled_training_dataset_x = mms.transform(training_dataset_x)
scaled_test_dataset_x = mms.transform(test_dataset_x)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='Auto-MPG')
model.add(Dense(64, activation='relu', input_dim=training_dataset_x.shape[1], name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae']) #Loss mse ise metrik olarak mae yazabiliriz.
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, epochs=200, validation_split=0.2)

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
plt.title('Epoch-Mean Absolute Error Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 210, 10))

plt.plot(hist.epoch, hist.history['mae'])
plt.plot(hist.epoch, hist.history['val_mae'])
plt.legend(['Mean Absolute Error', 'Validation Mean Absolute Error'])
plt.show()

eval_result = model.evaluate(scaled_test_dataset_x, test_dataset_y)

#mae 2 ise mesela gerçek değer 18 ise 16 veya 20 diye tahmin eder demektir.

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
 
import numpy as np

predict_data = np.array([[4, 91, 81, 2230, 17.5, 71, 1, 0, 0], [4, 150, 92, 2164, 16.5, 70, 0, 1, 1], [3, 111, 90, 2428, 15, 78, 0, 0, 1]])

scaled_predict_data = mms.transform(predict_data)

predict_result = model.predict(scaled_predict_data)

for val in predict_result[:, 0]:
    print(val)

model.save('auto-mpg.h5')

import pickle

with open('auto-mpg.pickle', 'wb') as f:
    pickle.dump(mms, f)
