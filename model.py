import keras
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D

def generator_model():
	model = Sequential()
	model.add(Dense(32, input_dim = 100))
	model.add(Activation('sigmoid'))
	model.add(Dense(64))
	model.add(Activation('relu'))

	return model

def discriminator_model():
	model = Sequential()
	return model

def adversial_model(generator, discriminator):
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	return model

