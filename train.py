import keras
import cv2

import numpy as np

from load import load_args
from load import load_img

def train(PATH, batch, epochs):
	img = load_img(PATH)
	for epoch in range(epochs):
		if(epoch == 0):
				if os.path.exist('generator_weights') ans os.path.exist('discriminator_weights'):
					print "Loading Saved Weights.."
					generator.load_weights('generator_weights')
					discriminator.load_weights('discriminator_weights')
					print "Finished Loading"
					pass

	
if __name__ == '__main__':
	args = load_args()	
	print args

	train(args.PATH, args.batch, args.epochs)

