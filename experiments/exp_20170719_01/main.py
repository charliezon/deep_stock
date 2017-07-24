import sys
sys.path.append('../../')

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
import h5py

from utils.metrics import precision

# Accuracy on test set: 64%

data = pd.read_csv("../../data/data_20170719_01/data.csv", header=None)
dataset = data.values

feature_len = 100
train_len = int(len(dataset)*0.96)
epochs =  1000
#epochs = 1000
num_unit = 128
#num_unit = 256
batch_size = 128
#batch_size = 256
#num_layer = 6
num_layer = 5
dropout = 0.5

x_train = dataset[0:train_len, 0:feature_len].astype(float)
y_train = dataset[0:train_len, feature_len]
x_test = dataset[train_len:, 0:feature_len].astype(float)
y_test = dataset[train_len:, feature_len]

#leakyReLU = LeakyReLU(alpha=0.3)

model = Sequential()
model.add(Dense(num_unit, input_dim=feature_len))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(dropout))

for i in range(num_layer):
    model.add(Dense(num_unit))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(dropout))

model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy', precision])
# model.load_weights('./model_weights.h5')
history = model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1,
          validation_split=0.1)

model.save_weights('./model_weights.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy, precision and loss')
plt.ylabel('accuracy, precision or loss')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc', 'train_precision', 'val_precision', 'train_loss', 'val_loss'], loc='upper left')
plt.show()

score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print(score)
