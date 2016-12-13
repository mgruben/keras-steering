# For saving / loading processed datasets
import pickle

# For feeding the data to the Keras model
import numpy as np

# For fixing silly JSON-isms
import json

# To simplify the TensorFlow interface,
# especially this will be a simple, normal model
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Conv2D
from keras.layers import MaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint

# Read in our training data
# This WILL take a while
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')

# Define a sequential model
# 
# We can imagine that subsequent models might employ a three-headed
# neural network: a center, a right, and a left, which all merge
# prior to some fully-connected layers who form a consensus.
# 
# But here, let's start simple.
model = Sequential()

# A Convolutional Layer with maxpooling, dropout, and ReLU
# 
# Note that, after resizing, the input layer is 3 channels at 32 x 16.
# 
# Let's make use of that by applying a 4 x 4 max pooling layer, rather
# than the standard 2 x 2.
# 
model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(32, 16, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# A Convolutional Layer with maxpooling, dropout, and ReLU
model.add(Conv2D(16, 3, 3, border_mode='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# Flatten before passing to fully-connected layers
model.add(Flatten())

# A fully-connected layer with dropout and ReLU
model.add(Dense(200, init='normal'))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# A fully-connected layer with dropout and ReLU
model.add(Dense(100, init='normal'))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# Our output layer.  Outputs steering angle.
model.add(Dense(1))

# Define our loss, optimizer, and metrics
model.summary()
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

# Checkpoints after each epoch
checkpointer = ModelCheckpoint(filepath="checkpoint.log", verbose=1, save_best_only=False)

# View the model training. eee!
history = model.fit(X_train, Y_train, batch_size=350, nb_epoch=20,
                    verbose=1)

# Save the output of our model, as requested
model_json = model.to_json()
with open('model.json', 'w') as f:
    json.dump(model_json, f)

model.save_weights('model.h5')

# Also save our history to a pickle file, for later perusing
with open('history.p', 'wb') as f:
    pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)
