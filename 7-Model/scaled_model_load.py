from tensorflow.keras.models import load_model

model = load_model('diabetes_mms.h5')

import pickle

with open('diabetes.pickle', 'rb') as f:
    mms = pickle.load(f)

import numpy as np    

predict_data = np.array([[2,148,58,37,128,25.4,0.699,24], [7,114,67,0,0,32.8,0.258,42],
[5,99,79,27,0,29,0.203,32], [0,110,85,30,0,39.5,0.855,39]]) 

transformed_predict_data = mms.transform(predict_data)
predict_result = model.predict(transformed_predict_data)

for i in range(len(predict_result)):
    print(predict_result[i, 0])
    
for i in range(len(predict_result)):
    print('Şeker Hastası' if predict_result[i, 0] > 0.5 else "Şeker Hastası Değil")
