import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense


X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
X = np.asarray(X)
y = np.asarray(y)

X = X/255.0

X_train = X[100:]
y_train = y[100:]
X_test = X[:100]
y_test = y[:100]

print(X.shape)

model = Sequential([
    Conv2D(128, (3,3), input_shape=X.shape[1:]),
    Activation("relu"),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3)),
    Activation("relu"),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3)),
    Activation("relu"),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(1),
    Activation("sigmoid")
])

model.compile(optimizer = "adam", 
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ["accuracy"])

model.fit(X_train, y_train, epochs=8, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("\nTest accuracy: ", test_acc)