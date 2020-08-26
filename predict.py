import sys
import cv2
import numpy as np
import tensorflow as tf

IMG_PATH = sys.argv[1]
IMG_SIZE = 50

img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img/255
img = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model('./saved_model')

pred = model.predict(img)
pred = float(pred)
pred = round(pred)

classes = ['Dog', 'Cat']
print(classes[pred])