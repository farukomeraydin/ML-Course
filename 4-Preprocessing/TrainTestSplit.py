import pandas as pd

TRAINING_RATIO = 0.8 

df = pd.read_csv('diabetes.csv')
dataset = df.to_numpy()

import numpy as np

np.random.shuffle(dataset)

training_zone = int(np.round(len(dataset) * TRAINING_RATIO))

training_dataset = dataset[:training_zone, :]
test_dataset = dataset[training_zone:, :] 

training_dataset_x = training_dataset[:, :-1]
training_dataset_y = training_dataset[:, -1]

test_dataset_x = test_dataset[:, :-1]
test_y = test_dataset[:, -1] 
