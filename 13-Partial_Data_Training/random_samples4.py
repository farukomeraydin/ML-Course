import numpy as np

EPOCHS = 10
BATCH_SIZE = 32

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, number_of_batches):
        super().__init__()
        self.number_of_batches = number_of_batches
        
    def __len__(self):
        return self.number_of_batches
    
    def __getitem__(self, index):
        x = np.random.random((BATCH_SIZE, 10))
        y = np.random.randint(0, 2, BATCH_SIZE)
        
        return x, y
    
    
    def on_epoch_end(self):
        print('on_epoch_end')
    

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='PartialDataTraining')
model.add(Dense(64, activation='relu', input_dim=10, name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(DataGenerator(5), epochs=EPOCHS, validation_data=DataGenerator(2)) #1 epoch 5 tane batchden oluşsun. Validation ise 2 parça verildi.

eval_result = model.evaluate(DataGenerator(20)) #Test işlemini 20 batch'de yaptık.

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
