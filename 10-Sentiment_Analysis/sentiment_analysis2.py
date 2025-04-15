import pandas as pd
import regex
import numpy as np

class WordCount:
    def __init__(self):
        self.word_set = set()
    
    def __call__(self, text):
        words = regex.findall("[A-Za-z-0-9'-]+", text.lower())
        self.word_set.update(words)
        
        return np.array(words) #np arrayleri daha az yer kaplar
        
    def get_word_dict(self):
        word_dict = {}
        for index, word in enumerate(self.word_set):
            word_dict[word] = index
            
        return word_dict

wc = WordCount()

df = pd.read_csv('IMDB Dataset.csv', converters={0: wc}) #Dosyayı okurken kelimelere ayıracağız

word_dict = wc.get_word_dict()


dataset_x = np.zeros((df.shape[0], len(word_dict)), dtype='uint8')

for row, words in enumerate(df['review']):
    word_indices = [word_dict[word] for word in words]
    dataset_x[row, word_indices] = 1
    
dataset_y = np.zeros(len(df), dtype='int8')
dataset_y[df['sentiment'] == 'positive'] = 1


from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='IMDB')
model.add(Dense(128, activation='relu', input_dim=dataset_x.shape[1], name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))

model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=1, validation_split=0.2)

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
    
texts = ['the movie was very good. The actors played perfectly. I would recommend it to everyone', 'this film is awful. The worst film i have ever seen'] 

for predict_text in texts:
    words = regex.findall("[A-Za-z-0-9'-]+", predict_text.lower())

    predict_vect = np.zeros(len(word_dict), dtype='int8')
    word_indices = [word_dict[word] for word in words]
    predict_vect[word_indices] = 1

    predict_result = model.predict(predict_vect.reshape(1, -1)) 
    
    if predict_result[0, 0] > 0.5:
        print('Positive')
    else:
        print('Negative')
        
model.save('imdb.h5')
