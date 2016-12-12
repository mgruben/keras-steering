# For saving / loading processed datasets
import pickle

# For feeding the data to the Keras model
import numpy as np

# To simplify the TensorFlow interface,
# especially this will be a simple, normal model
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Conv2D, Flatten


# Define a sequential model
# 
# We can imagine that subsequent models might employ a three-headed
# neural network: a center, a right, and a left, which all merge
# prior to some fully-connected layers who form a consensus.
# 
# But here, let's start simple.
model = Sequential()

# A Convolutional Layer with maxpooling, dropout, and ReLU
model.add(Conv2D(32, 3, 3, input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# A Convolutional Layer with maxpooling, dropout, and ReLU
model.add(Conv2D(32, 3, 3))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# Flatten before passing to fully-connected layers
model.add(Flatten())

# A fully-connected layer with dropout and ReLU
model.add(Dense(128, init='normal')
model.add(Dropout(0.5))
model.add(Activation('relu'))

# A fully-connected layer with dropout and ReLU
model.add(Dense(128, init='normal')
model.add(Dropout(0.5))
model.add(Activation('relu'))

# Our output layer.  Outputs steering angle.
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=128, nb_epoch=20,
                    verbose=1)
