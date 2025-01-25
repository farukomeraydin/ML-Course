from tensorflow.keras.models import load_model

model = load_model('auto-mpg.h5')

import pickle

with open('auto-mpg.pickle', 'rb') as f:
    mms = pickle.load(f)
    
import pandas as pd

df = pd.read_csv('predict-data.csv', header=None)
predict_data = pd.get_dummies(df, columns=[6]).to_numpy()

scaled_predict_data = mms.transform(predict_data)

predict_result = model.predict(scaled_predict_data)

for val in predict_result[:, 0]:
    print(val)
