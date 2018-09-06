import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import random

IMG_SIZE = 50

# callbacks
checkpoint = ModelCheckpoint("CNN_weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# loading data from *.pickle files
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0 # normalize data

model = Sequential()

model.add(Conv2D( 128, (3,3), input_shape = X.shape[1:] ))
model.add(Activation("relu"))
model.add(MaxPooling2D( pool_size=(2,2) ))

model.add(Conv2D( 128, (3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D( pool_size=(2,2) ))

model.add(Conv2D( 128, (3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D( pool_size=(2,2) ))

model.add(Flatten())
#model.add(Dense(64)) useless
#model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

try:
    model.load_weights('CNN_weights.hdf5')
    print('Weights loaded')
except:
    pass

training_data = []

# the model will be training till we end the program
while(True):
    model.fit(X, y, batch_size=32, epochs=8, validation_split=0.10, callbacks=callbacks_list) # training

    # reshuffle dataset. I don't even know if it helps
    for i in range(len(X)):
        training_data.append( [ X[i], y[i] ] )
    
    random.shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)