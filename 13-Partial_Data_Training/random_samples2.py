import numpy as np

EPOCHS = 10
STEPS_PER_EPOCH = 20
BATCH_SIZE = 32
VALIDATION_STEPS = 5

def data_generator():
    for i in range(EPOCHS):
        for k in range(STEPS_PER_EPOCH):
            x = np.random.random((BATCH_SIZE, 10))
            y = np.random.randint(0, 2, BATCH_SIZE, dtype='int8')
            
            yield x, y
            
def validation_generator():
    for i in range(EPOCHS):
        for k in range(VALIDATION_STEPS + 1): #Neden 1 fazla olmalı henüz bilmiyoruz.
            x = np.random.random((BATCH_SIZE, 10))
            y = np.random.randint(0, 2, BATCH_SIZE, dtype='int8')
            yield x, y
    

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='PartialDataTraining')
model.add(Dense(64, activation='relu', input_dim=10, name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(data_generator(), epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=validation_generator(), validation_steps=VALIDATION_STEPS)
