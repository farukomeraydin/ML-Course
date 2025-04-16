import numpy as np

EPOCHS = 100
STEPS_PER_EPOCH = 20
BATCH_SIZE = 32


def data_generator():
    for _ in range(EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            x = np.random.random((BATCH_SIZE, 10))
            y = np.random.randint(0, 2, (BATCH_SIZE, 1), dtype='int8')
            
            yield x, y

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='PartialDataTraining')
model.add(Dense(64, activation='relu', input_dim=10, name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(data_generator(), epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH) #Bu sefer y verisi girilmez.batch_size ve validation_split'e de gerek yok.
