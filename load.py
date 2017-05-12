import os
import glob
import argparse

import cv2

def load_img(path):
	img = cv2.imread(path, 1)

def load_args():
	parser = argparse.ArgumentParser(description = 'train the GAN model')
	parser.add_argument('--PATH', type = str, help = 'The path of training data')
	parser.add_argument('--batch', type = int, default = 50, help = 'batch size')
	parser.add_argument('--epochs', type = int, default = 2, help = 'number of epochs')

	return parser.parse_args()

