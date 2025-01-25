#Tek etiketli çok sınıflı bir sınıflandırma problemidir. Veri kümesinin azlığından ve doğasından dolayı genellikle accuracy yüksek çıkar.
import pandas as pd

df = pd.read_csv('iris.csv')

dataset_x = df.iloc[:, 1:-1].to_numpy(dtype='float32')

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
dataset_y = ohe.fit_transform(df.iloc[:, -1].to_numpy().reshape(-1, 1)) #unique yapmak zorunda olmamak için get_dummies yerine bu sınıfı kullandık.

#dataset_y = pd.get_dummies(df.iloc[:, -1]).to_numpy(dtype='int8') #ohe yapıldı.
from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x) 
scaled_test_dataset_x = ss.transform(test_dataset_x) 

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='Iris')
model.add(Dense(64, activation='relu', input_dim=training_dataset_x.shape[1], name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(3, activation='softmax', name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
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
plt.title('Epoch-Categorical Accuracy Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 210, 10))

plt.plot(hist.epoch, hist.history['categorical_accuracy'])
plt.plot(hist.epoch, hist.history['val_categorical_accuracy'])
plt.legend(['Categorical Accuracy', 'Validation Categorical Accuracy'])
plt.show()

eval_result = model.evaluate(scaled_test_dataset_x, test_dataset_y) 

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

predict_df = pd.read_csv('predict-iris.csv', header=None) #İlk satırı header zannetmesin
predict_data = predict_df.to_numpy(dtype='float32')

scaled_predict_data = ss.transform(predict_data)
predict_result = model.predict(scaled_predict_data)

import numpy as np
result_index = np.argmax(predict_result, axis=1) #Sınıflar lexikografik sıraya dizilip yerini alır.0:iris-setosa, 1:iris-versicolor, 2:iris-virginica

result_names = ohe.categories_[0][result_index]
print(result_names)
#predict_result_names = df.iloc[:, -1].unique()[np.argmax(predict_result, axis=1)]#get_dummies kullanırsak böyle

model.save('iris.h5')

import pickle

with open('iris.pickle', 'wb') as f:
    pickle.dump(ss, f)
