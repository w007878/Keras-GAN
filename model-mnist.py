import keras
import numpy as np
import cv2
import os
import argparse

from keras.models import Sequential, Model
from keras.layers import Dense
from keras.datasets import mnist

def load_args():
	parser = argparse.ArgumentParser(description = 'the GAN model')
	parser.add_argument('--type', type = str, choices = ['train', 'generate'], default = 'train', help = 'train or generate')
	parser.add_argument("--batch_size", type = int, default = 50)
	parser.add_argument("--epochs", type = int, default = 2)
	parser.add_argument("--img_num", type = int, default = 10)
	return parser.parse_args()

def generator_model(inputdim = 100):
	model = Sequential()
	model.add(Dense(32, input_dim = inputdim, activation = 'relu'))
	model.add(Dense(16, activation = 'tanh'))
	model.add(Dense(32, activation = 'relu'))
	model.add(Dense(128, activation = 'tanh'))
	model.add(Dense(784, activation = 'sigmoid'))
	return model

def discriminator_model():
	model = Sequential()
	model.add(Dense(128, input_dim = 784, activation = 'relu'))
	model.add(Dense(64, activation = 'sigmoid'))
	model.add(Dense(32, activation = 'relu'))
	model.add(Dense(32, activation = 'sigmoid'))
	model.add(Dense(16, activation = 'tanh'))
	model.add(Dense(1, activation = 'sigmoid'))
	return model

def adversial_model(generator, discriminator):
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	return model

def chunks(l, n):
	for i in xrange(0, len(l), n):
		yield l[i:i + n]

def generate_noise(dim):
	return np.random.uniform(-1, 1, dim)

def train(epochs, batch_size):
	(x_train, _), (x_test, _) = mnist.load_data()

	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.

	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

	img_batchs = [q for q in chunks(x_train, batch_size)]

	generator = generator_model()
	discriminator = discriminator_model()
	adversial = adversial_model(generator, discriminator)

	generator.compile(optimizer = 'sgd', loss = 'binary_crossentropy')
	discriminator.compile(optimizer = 'sgd', loss = 'binary_crossentropy')
	adversial.compile(optimizer = 'sgd', loss = 'binary_crossentropy')

	for epoch in range(epochs):
		
		print 'Epoch', epoch

		if epoch == 0:
			if os.path.exists('generator_weights') and os.path.exists('discriminator_weights'):
				print 'Loading Saved Weights'
				generator.load_weights('generator_weights')
				discriminator.load_weights('discriminator_weights')

		for index, img_batch in enumerate(img_batchs):
			print 'Epoch', epoch, 'Batch', index
			
			noise_batch = np.array([generate_noise(100) for i in range(len(img_batch))])

			generated_img = generator.predict(noise_batch)

			Xd = np.concatenate((img_batch, generated_img))
			Yd = np.concatenate(([1] * len(img_batch), [0] * len(generated_img)))

			discriminator.train_on_batch(Xd, Yd)

			Xg = noise_batch
			Yg = np.array([1] * len(noise_batch))
			adversial.train_on_batch(Xg, Yg)

			if index % 10 == 0:
				print 'Saving Weights'
				generator.save_weights('generator_weights', True)
				discriminator.save_weights('discriminator_weights', True)
		
		print 'Saving Weights Each Epoch'
		generator.save_weights('generator_weights', True)
		discriminator.save_weights('discriminator_weights', True)

def generate(img_num):
	generator = generator_model()
	generator.compile(optimizer = 'sgd', loss = 'binary_crossentropy')
	generator.load_weights('generator_weights')

	if not os.path.exists('results'):
		os.mkdir('results')

	
	generated_img = []
	for i in range(img_num):
		noise = np.array([generate_noise(100) for n in range(img_num)])

		tmp_img  = np.concatenate([(255 * img.reshape(28, 28)).astype('int') for img in generator.predict(noise)])

		if len(generated_img) == 0:
			generated_img = tmp_img
		else:
			generated_img = np.concatenate([generated_img, tmp_img], 1)

		print generated_img.shape

#	tmp_img = generated_img.reshape((28, len(generated_img) * 28))
		cv2.imwrite('results/results_large.jpg', generated_img)

#	for index, img in enumerate(generated_img):
#		cv2.imwrite('results/' + '{}.jpg'.format(index), img)


if __name__ == '__main__':
	args = load_args()
	if(args.type == 'train'):
		train(args.epochs, args.batch_size)
	else:
		generate(args.img_num)

