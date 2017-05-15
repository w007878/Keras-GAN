import keras
import numpy as np
import os

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

encoding_dim = 32

input_img = Input(shape = (784, ))

encoded = Dense(encoding_dim, activation = 'relu')(input_img)
decoded = Dense(784, activation = 'sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape = (encoding_dim, ))

decoder_layer = autoencoder.layers[-1]

decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer = 'sgd', loss = 'binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

print x_train.shape
print np.prod(x_train.shape[1:])
print np.prod(x_train.shape)

print encoding_dim 



