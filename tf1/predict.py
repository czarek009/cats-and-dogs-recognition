import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import sys
import cv2

file_name = sys.argv[1]

IMG_SIZE = 50 # You can't just change this variable. 

img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

img = img/255

img = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = Sequential()

model.add(Conv2D( 64, (3,3), input_shape = (img.shape[1:]) ))
model.add(Activation("relu"))
model.add(MaxPooling2D( pool_size=(2,2) ))

model.add(Conv2D( 64, (3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D( pool_size=(2,2) ))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

filepath="CNN_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

try:
    model.load_weights('CNN_weights.hdf5')
    print('Weights loaded')

    ans = model.predict(img)
    ans = float(ans)
    ans = round(ans)

    if ans == 1:
        print('Cat')
    elif ans == 0:
        print('Dog')
    else:
        print('Wtf, idk what it is')
except:
    print('I need weights')