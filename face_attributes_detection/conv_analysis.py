import os
import sys
import math
import getopt
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Activation, Dense, Flatten, Input, \
    BatchNormalization, Lambda
import platform

host = platform.node()

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

images_path = root_path + "cropped_images/"

model = tf.keras.models.load_model(root_path + "facenet.h5")
model.summary()
# load the image with the required shape
path = images_path + '000007.jpg'
im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY).astype('int32').reshape((-1,36,36,1))
# get feature map for first hidden layer
"""feature_maps = model.predict(im)

# plot all 16 maps in an 8x8 squares
square = 4
ix = 1
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
plt.show()"""