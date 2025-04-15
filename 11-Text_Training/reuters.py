"""
ReutersData 
	--training
	--test
	--cats.txt
"""
import os

training_label_dict = {}
test_label_dict = {}
with open('ReutersData/cats.txt') as f:
    for line in f:
        a = line.split()
        label_type, pos = a[0].split('/')
        pos = int(pos[pos.find('/') + 1:])
        if label_type == 'training':
            training_label_dict[pos] = a[1]
        elif label_type == 'test':
            test_label_dict[pos] = a[1]
        """
        VEYA
        pos_text = a[0]
        pos = pos_text.find('/')
        pos_val = int(pos_text[pos + 1:])
        if pos == 8:
            training_label_dict[pos_val] = a[1]
        else:
            test_label_dict[pos_val] = a[1]
        """
all_labels_set = set(training_label_dict.values())
all_labels_set.update(test_label_dict.values()) #test ve traindeki etiket kümelerini birleştirdik
all_labels_list = list(all_labels_set)
        
training_dataset_ylist = [] 
reuters_training_text = []

for fname in os.listdir('ReutersData/training'):
    with open('ReutersData/training/' + fname) as f:
        reuters_training_text.append(f.read()) #Dosyaları okuyup listeye ekleriz
    val = training_label_dict[int(fname)]
    index = all_labels_list.index(val)
    training_dataset_ylist.append(index)
    
test_dataset_ylist = []         
reuters_test_text = []
for fname in os.listdir('ReutersData/test'):
    with open('ReutersData/test/' + fname, encoding='latin-1') as f: #UTF-8 ile yapılmayan dosya vardı
        reuters_test_text.append(f.read())
        val = test_label_dict[int(fname)]
        index = all_labels_list.index(val)
        test_dataset_ylist.append(index)

import numpy as np        
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, categories=[range(len(all_labels_list))]) #bazı labellar trainingde bazıları ise testte olabilir. Onun için categories parametresi gerekli.
training_dataset_y = ohe.fit_transform(np.array(training_dataset_ylist).reshape(-1, 1))
test_dataset_y = ohe.fit_transform(np.array(test_dataset_ylist).reshape(-1, 1))
        
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
cv.fit(reuters_training_text + reuters_test_text)
training_dataset_x = cv.transform(reuters_training_text).todense()
test_dataset_x = cv.transform(reuters_test_text).todense()

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='Reuters')
model.add(Dense(64, activation='relu', input_dim=training_dataset_x.shape[1], name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(len(all_labels_list), activation='softmax', name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=5, validation_split=0.2)
#Overfitting olmuş çünkü accuracy val_accuracyden uzaklaşmış. Onun için epoch az seçilmeli.
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')

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

eval_result = model.evaluate(test_dataset_x, test_dataset_y)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')


texts = ['countries trade volume is expanding. Trade is very important for countries.', 'It is known that drinking a lot of coffee is harmful to health. However, according to some, a little coffee is beneficial.']

predict_data = cv.transform(texts).todense()

predict_result = model.predict(predict_data)
predict_result = np.argmax(predict_result, axis=1)

for pr in predict_result:
    print(all_labels_list[pr])
