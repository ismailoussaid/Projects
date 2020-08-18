# Libraries importation
import cv2
import os
import glob
import pandas as pd

# Loading of images
global_path = "C:/Users/Ismail/Documents/Projects/celeba_files/celeba_img"
final_path = "C:/Users/Ismail/Documents/Projects/celeba_files/cropped_images"

def create_folder(path):
    try:
        os.makedirs(path)
        print("<== Directory created ==>")
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)

# Image Processing
def crop_dataset():

    filenames = glob.glob(global_path+'/*.jpg')
    tab = pd.read_csv("C:/Users/Ismail/Documents/Projects/celeba_files/celeba_csv/list_landmarks_align_celeba.csv")

    create_folder(final_path)

    epsilon = 0.5
    grand = 1 + epsilon
    small = 1 - epsilon

    for k in range(len(filenames)):

        compteur = k+1
        box = tab.iloc[k]
        resize_img_path = '{}/{}'.format(final_path, box.image_id)
        img_path = global_path+'/'+box.image_id
        img = cv2.imread(img_path, 0)

        if type(img) != type(None):

            crop_img = img[int(box.nose_y * (small - 0.2)):int(box.nose_y * (grand - 0.2)),
                   int(box.nose_x * small):int(box.nose_x * grand)]

            if crop_img != []: # ensure image is not empty
                resize_img = cv2.resize(crop_img, dsize=(36, 36))
                cv2.imwrite(resize_img_path, resize_img)

        if compteur %5000 == 0:
            print("{} images out of {} have been cropped".format(compteur, tab.shape[0]))

    print("All functional images have been cropped and put in the new directory")

#crop_dataset()

# Building datasets for each attribute
def build_dataset(attribute, positive_attribute, negative_attribute):

    filenames = glob.glob(final_path+'/*.jpg')
    tab = pd.read_csv("C:/Users/Ismail/Documents/Projects/celeba_files/celeba_csv/list_attr_celeba.csv")
    label_path = [
        "C:/Users/Ismail/Documents/Projects/celeba_files/{}/train/{}".format(attribute, positive_attribute.lower()),
        "C:/Users/Ismail/Documents/Projects/celeba_files/{}/train/{}".format(attribute, negative_attribute.lower()),
        "C:/Users/Ismail/Documents/Projects/celeba_files/{}/val/{}".format(attribute, positive_attribute.lower()),
        "C:/Users/Ismail/Documents/Projects/celeba_files/{}/val/{}".format(attribute, negative_attribute.lower()),
        "C:/Users/Ismail/Documents/Projects/celeba_files/{}/test/{}".format(attribute, positive_attribute.lower()),
        "C:/Users/Ismail/Documents/Projects/celeba_files/{}/test/{}".format(attribute, negative_attribute.lower())
    ]

    for path in label_path:
        create_folder(path)

    m = len(filenames)

    for k in range(0, m):

        compteur = k+1
        box = tab.iloc[k]
        label = box[positive_attribute]

        if k < int(0.8*m):
            if label == 1:
                img_path = '{}/{}'.format(label_path[0], box.image_id)
            else:
                img_path = '{}/{}'.format(label_path[1], box.image_id)
        elif k < int(0.9*m):
            if label == 1:
                img_path = '{}/{}'.format(label_path[2], box.image_id)
            else:
                img_path = '{}/{}'.format(label_path[3], box.image_id)
        else:
            if label == 1:
                img_path = '{}/{}'.format(label_path[4], box.image_id)
            else:
                img_path = '{}/{}'.format(label_path[5], box.image_id)

        img = cv2.imread('{}/{}'.format(final_path, box.image_id), 0)
        cv2.imwrite(img_path, img)

        if compteur %5000 == 0:
            print("{} images out of {} have been put in {} dataset".format(compteur, tab.shape[0], attribute))

    print("All functional images have been  put in the new directory")

build_dataset('beard', 'No_Beard', 'Beard')
build_dataset('eyeglasses', 'Eyeglasses', 'Naked_eye')
build_dataset('hat', 'Wearing_Hat', 'No_Hat')
build_dataset('gender', 'Male', 'Female')
build_dataset('hairstyle', 'Bald', 'Hairy')
build_dataset('mustache', 'Mustache', 'No_Mustache')