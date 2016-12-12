# For saving / loading processed datasets
import pickle

# For feeding the data to the Keras model
import numpy as np

# To simplify the TensorFlow interface,
# especially this will be a simple, normal model
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Conv2D, Flatten



model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2,2)))
model.add((Dropout(0.5)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(43, activation='softmax'))

model.summary()
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=128, nb_epoch=20,
                    verbose=1)
