import pandas as pd

df = pd.read_csv('diabetes.csv')

dataset_x = df.iloc[:, :-1].to_numpy()
dataset_y = df.iloc[:, -1].to_numpy()

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='Sample')
model.add(Dense(32, activation='relu', input_dim=dataset_x.shape[1], name='Hidden-1'))
model.add(Dense(32, activation='relu', name='Hidden-2')) 
model.add(Dense(1, activation='sigmoid', name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=200, validation_split=0.2)

hidden1 = model.layers[0]
hidden1_weights = hidden1.get_weights()

print(type(hidden1_weights))
print(len(hidden1_weights))
print(type(hidden1_weights[0]), type(hidden1_weights[1]))
print(hidden1_weights[0].shape, hidden1_weights[1].shape)

w = hidden1_weights[0][5, 17]
print(w)
