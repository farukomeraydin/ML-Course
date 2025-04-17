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

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, x_df, y, batch_size, shuffle=True):
        super().__init__()
        self.x_df = x_df
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.x_df) // self.batch_size))
    
    def __len__(self):
        return len(self.x_df) // self.batch_size
    
    def __getitem__(self, index):
        start_index = self.indices[index] * self.batch_size
        stop_index = (self.indices[index] + 1) * self.batch_size
        
        x = cv.transform(self.x_df.iloc[start_index:stop_index]).todense() #Benden istediği kısmı vektörize ediyorum.
        y = self.y[start_index:stop_index]
            
        return x, y
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='IMDB')
model.add(Dense(64, activation='relu', input_dim=len(cv.vocabulary_), name='Hidden-1')) 
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output')) 

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

hist = model.fit(DataGenerator(training_df_x, training_y, BATCH_SIZE), validation_data=DataGenerator(validation_df_x, validation_y, BATCH_SIZE, False), epochs=EPOCHS) 
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

eval_result = model.evaluate(DataGenerator(test_df_x, test_y, BATCH_SIZE, False))
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
