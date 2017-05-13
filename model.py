import keras
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

def generater_model():
	model = Sequential()
	model.add(Dense(32, input_dim = 32 * 32))
	return model

def discriminator_model():
	model = Sequential()
	return model

