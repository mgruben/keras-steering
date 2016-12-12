# For saving / loading processed datasets
import pickle

# For feeding the data to the Keras model
import numpy as np

# To simplify the TensorFlow interface,
# especially this will be a simple, normal model
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Conv2D
from keras.layers import MaxPooling2D, Flatten, Dropout


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
# Note that the input layer is 3 channels at 160 x 320, so we have
# plenty of extra information to pare down without suffering much
# of an accuracy loss.
# 
# Let's make use of that by applying a 4 x 4 max pooling layer, rather
# than the standard 2 x 2.
# 
model.add(Conv2D(64, 5, 5, border_mode='same', input_shape=(160, 320, 3)))
model.add(MaxPooling2D((4,4)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# A Convolutional Layer with maxpooling, dropout, and ReLU
model.add(Conv2D(32, 5, 5, border_mode='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# Flatten before passing to fully-connected layers
model.add(Flatten())

# A fully-connected layer with dropout and ReLU
model.add(Dense(256, init='normal'))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# A fully-connected layer with dropout and ReLU
model.add(Dense(128, init='normal'))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# Our output layer.  Outputs steering angle.
model.add(Dense(1))

# View the model training. eee!
model.summary()
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=128, nb_epoch=20,
                    verbose=1)

# Save the output of our model, as requested
with open('model.json', 'w') as f:
    f.write(model.to_json())

model.save_weights('model.h5')

# Also save our history to a pickle file, for later perusing
with open('history.p', 'wb') as f:
    f.write(history.history, f, pickle.HIGHEST_PROTOCOL)
