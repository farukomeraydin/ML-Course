from tensorflow.keras.models import load_model

model = load_model('auto-mpg.h5')

import pickle

with open('auto-mpg.pickle', 'rb') as f:
    mms = pickle.load(f)

import pandas as pd

df = pd.read_csv('predict-data-2.csv', header=None)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, categories=[[1, 2, 3]]) # csv'de 2 tane kategori varsa 2'li ohe yapar ama 3 olmalı çünkü 3 sınıf var

df[['ohe1', 'ohe2', 'ohe3']] = ohe.fit_transform(df.iloc[:, 6].to_numpy(dtype='float32').reshape(-1, 1)) #2-D girdi istediği için

df.drop(6, axis=1, inplace=True) #6.sütun drop edildi

predict_data = df.to_numpy(dtype='float32')
scaled_predict_data = mms.transform(predict_data)

predict_result = model.predict(scaled_predict_data)

for val in predict_result[:, 0]:
    print(val)
