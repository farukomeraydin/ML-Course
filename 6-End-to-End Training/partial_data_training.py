EPOCHS = 5
BATCH_SIZE = 32

import pandas as pd

df = pd.read_csv('IMDB Dataset.csv')

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(dtype='int8')
cv.fit(df['review'])

df_x = df['review']
df_y = df['sentiment']

import numpy as np

dataset_y = np.zeros(len(df_y), dtype='int8')
dataset_y[df['sentiment'] == 'positive'] = 1

from sklearn.model_selection import train_test_split

temp_df_x, test_df_x, temp_y, test_y = train_test_split(df_x, dataset_y, test_size=0.20)

training_df_x, validation_df_x, training_y, validation_y = train_test_split(temp_df_x, temp_y, test_size=0.20)

def data_generator(x_df, y_df, steps, shuffle=True): #df_x ve df_y parametreleriyle genelleştirdik.Yani hem validation hem training verilerini generate ediyor
    indices = list(range(steps))
    for _ in range(EPOCHS):
        if shuffle:
            np.random.shuffle(indices) #karıştırıldı
        for i in range(steps):
            start_index = indices[i] * BATCH_SIZE
            stop_index = (indices[i] + 1) * BATCH_SIZE
            #Şimdilik shuffle yapmıyoruz
            x = cv.transform(x_df.iloc[start_index:stop_index]).todense() #Benden istediği kısmı vektörize ediyorum.
            y = y_df[start_index:stop_index]
            
            yield x, y

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='IMDB')
model.add(Dense(64, activation='relu', input_dim=len(cv.vocabulary_), name='Hidden-1')) 
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output')) 

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

steps_per_epoch = len(training_df_x) // BATCH_SIZE
steps_per_validation = len(validation_df_x) // BATCH_SIZE
steps_per_test = len(test_df_x) // BATCH_SIZE

hist = model.fit(data_generator(training_df_x, training_y, steps_per_epoch), steps_per_epoch=steps_per_epoch, validation_data=data_generator(validation_df_x, validation_y, steps_per_validation + 1, False), validation_steps=steps_per_validation, epochs=EPOCHS) #Validationda shuffle'a gerek olmadığı için False verdik.

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

eval_result = model.evaluate(data_generator(test_df_x, test_y, steps_per_test), steps=steps_per_test)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
