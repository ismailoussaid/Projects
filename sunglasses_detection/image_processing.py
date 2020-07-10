# Libraries importation
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import json

# Loading of images
global_path = "C:/Users/Ismail/Documents/QuividiData/"
sunglasses = "C:/Users/Ismail/Documents/QuividiData/sunglasses/"
no_sunglasses = "C:/Users/Ismail/Documents/QuividiData/img/"

paths_sunglasses = glob.glob(os.path.join(sunglasses, "*.jpg")) + glob.glob(os.path.join(sunglasses, "*.jpeg"))
paths_no_sunglasses = glob.glob(os.path.join(no_sunglasses, "*.jpg")) + glob.glob(os.path.join(no_sunglasses, "*.jpeg"))

path_dataset = paths_sunglasses + paths_no_sunglasses

# Image Processing
def crop_dataset(folder):
    # define the name of the directory to be created
    path = "C:/Users/Ismail/Documents/QuividiData/cropped_images"
    folders = [subdir for subdir in glob.glob(folder + "*/") if 'IDBTools' not in subdir] #get image data only
    folders = [subdir for subdir in folders if 'cropped' not in subdir]  #ensure to only get data from img and sunglasses
    compteur, data = 0, []

    for subdir in folders:
        with open('{}objects.json'.format(subdir), 'r') as f:
            set = json.load(f)
            data += set
    n = len(data)

    print("There are {} images overall".format(n))

    try:
        os.makedirs(path)
        print("<== Directory created ==>")
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)

    for subdirectory in folders:

        with open('{}objects.json'.format(subdirectory)) as f:
            dataset = json.load(f)

        for data in dataset:
            compteur += 1
            file = data['file']
            img_path = '{}{}'.format(subdirectory, file)
            box = data['bbox']
            x = max(0,box[0])
            y = max(1,box[1])
            w = box[2]
            h = box[3]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if type(img) != type(None):

                crop_img = img[y:y + h, x:x + w]  # crop the image

                if crop_img != []: # ensure image is not empty

                    if 'img' in subdirectory:
                        if "SUNGLASSES" in data['features']:
                            cv2.imwrite(path + '/sunglasses_' + str(compteur) + '.jpg', crop_img)
                        else:
                            cv2.imwrite(path + '/no_sunglasses_' + str(compteur) + '.jpg', crop_img)

                    else:
                        cv2.imwrite(path + '/sunglasses_' + str(compteur) + '.jpg', crop_img)

            if compteur %50 == 0:
                print("{} images out of {} have been cropped".format(compteur, n))

    print("All functional images have been cropped and put in the new directory")

crop_dataset(global_path)