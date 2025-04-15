from tensorflow.keras.datasets import imdb

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = imdb.load_data() #Sözcüklerin numaralarını verir.Vektörizasyonu sen yapacaksın.

vocab_dict=imdb.get_word_index()

rev_vocab_dict = {value: key for key, value in vocab_dict.items()}

def get_text(text_numbers):
    return ' '.join([rev_vocab_dict[number - 3] for number in text_numbers if number > 2]) #0, 1 ve 2 indekslerine özel başka bir şey konmuş.

text = get_text(training_dataset_x[0])
print(text)

import numpy as np

def vectorize(dataset, vocab_size):
    vect = np.zeros((len(dataset), vocab_size), dtype='int8')
    for index, vals in enumerate(dataset):
        vect[index, vals] = 1
        
    return vect

training_dataset_x = vectorize(training_dataset_x, len(vocab_dict) + 3)
test_dataset_x = vectorize(test_dataset_x, len(vocab_dict) + 3)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='IMDB')
model.add(Dense(128, activation='relu', input_dim=training_dataset_x.shape[1], name='Hidden-1')) 
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output')) 

model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=5, validation_split=0.2)

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

eval_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
