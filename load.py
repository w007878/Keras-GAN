import os
import glob
import argparse

import cv2
import numpy as np

def read_resize_image(path):
	img = cv2.imread(path, 1)
	img = cv2.resize(img, (32, 32))
	img = numpy.rollaxis(img, 2, 0)
	return img

def load_img(path):
	path = glob.glob(os.path.join(path, '*.jpg'))
	print path
	return np.array([read_resize_image(p) for p in path])

def load_args():
	parser = argparse.ArgumentParser(description = 'train the GAN model')
	parser.add_argument('--PATH', type = str, help = 'The path of training data')
	parser.add_argument('--batch', type = int, default = 50, help = 'batch size')
	parser.add_argument('--epochs', type = int, default = 2, help = 'number of epochs')

	return parser.parse_args()

