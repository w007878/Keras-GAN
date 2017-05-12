import keras
import cv2

from load import load_args
from load import load_img

if __name__ == '__main__':
	args = load_args()	
	print args

	img = load_img(args.PATH)
