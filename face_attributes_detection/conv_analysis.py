import os
import sys
import math
import getopt
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
import platform

host = platform.node()
aligned = True
root_linux = "/dev/shm/data/celeba_files/"
root_windows = "C:/Users/Ismail/Documents/Projects/celeba_files/"
root_scaleway = '/root/data/celeba_files/'

if host == 'castor' or host == 'altair':  # Enrico's PCs
    root_path = root_linux
elif host == 'DESKTOP-AS5V6C3':  # Ismail's PC
    root_path = root_windows
elif host == 'scw-zealous-ramanujan':
    root_path = root_scaleway
else:
    raise RuntimeError('Unknown host')

if aligned:
	images_path = root_path + "well_cropped_images/"
else:
	images_path = root_path + "cropped_images/"

model = tf.keras.models.load_model(root_path + "facenet_flops_test.h5")
model.summary()
# load the image with the required shape
path = images_path + '000414.jpg'
im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY).astype('int32').reshape((-1,36,36,1))
# get feature map for first hidden layer
second_conv = model.layers[5]
first_conv = model.layers[1]

# plot all 16 maps in an 8x8 squares
def save_feature_map(square, model_filename, number_layer, img_filename, filename="conv_2_analysis.jpg"):
	ix = 1
	path = images_path + img_filename
	model = tf.keras.models.load_model(root_path + model_filename)
	layer = model.layers[number_layer]
	model = Model(inputs=model.inputs, outputs=layer.output)
	feature_maps = model.predict(im)

	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = plt.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
			ix += 1
	# show the figure
	plt.savefig(root_path+filename)

if __name__ == '__main__':
	facenet_filename = "facenet_flops_test.h5"
	image_filename = '000414.jpg'
	save_feature_map(2, facenet_filename, 1, image_filename, filename="conv_1_analysis.jpg")
	save_feature_map(4, facenet_filename, 5, image_filename, filename="conv_2_analysis.jpg")