import numpy as np
import pandas as pd

df = pd.read_csv('diabetes.csv')

dataset_x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
dataset_y = df.iloc[:, -1].to_numpy(dtype=np.float32)

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
mms.fit(training_dataset_x)

transformed_training_dataset_x = mms.transform(training_dataset_x)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='Diabetes')
model.add(Dense(64, activation='relu', input_dim=dataset_x.shape[1], name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2')) 
model.add(Dense(1, activation='sigmoid', name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(transformed_training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2)


transformed_test_dataset_x = mms.transform(test_dataset_x)
eval_result = model.evaluate(transformed_test_dataset_x, test_dataset_y)

model.save('diabetes_mms.h5')
import pickle

with open('diabetes.pickle', 'wb') as f:
    pickle.dump(mms, f)
