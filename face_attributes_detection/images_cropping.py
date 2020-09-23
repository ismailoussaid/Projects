# Libraries importation
import cv2
import os
import glob
import pandas as pd
import platform

host = platform.node()

root_linux = "/dev/shm/data/celeba_files/"
root_windows = "C:/Users/Ismail/Documents/Projects/celeba_files/"
root_scaleway = '/root/data/celeba_files/'
aligned = True
multi_folder = False
crop = True

if host == 'castor' or host == 'altair':  # Enrico's PCs
    root_path = root_linux
elif host == 'DESKTOP-AS5V6C3':  # Ismail's PC
    root_path = root_windows
elif host == 'scw-zealous-ramanujan' or host == 'scw-cranky-jang':
    root_path = root_scaleway
else:
    raise RuntimeError('Unknown host')

# Loading of images
global_path = root_path + "celeba_img"
attributes_path = root_path + "celeba_csv"

if aligned:
    bbox_path = root_path + "celeba_csv/aligned.csv"
else:
    bbox_path = root_path + "celeba_csv/list_bbox_celeba.csv"

def create_folder(path):
    try:
        os.makedirs(path)
        print("<== Directory created ==>")
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)

# Image Processing with CelebA bbox
def crop_dataset(output_path="cropped_images"):

    filenames = glob.glob(global_path+'/*.jpg')
    tab = pd.read_csv(bbox_path)

    final_path = "C:/Users/Ismail/Documents/Projects/celeba_files/"+output_path
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

# Image Processing with Quividi bbox
def crop_well_dataset(alpha=0.7, beta=1.1, output_path="well_cropped_images"):

    new_attributes_path = attributes_path + "/list_attr_celeba_aligned.csv"
    tab_new = pd.read_csv(root_path + "celeba_csv/aligned.csv")
    tab_old = pd.read_csv(root_path + "celeba_csv/list_attr_celeba.csv")
    tab_new = tab_new.drop(["# ID", "bbox_x", "bbox_y", "bbox_w", "bbox_h"], axis=1)
    tab_join = pd.merge(left=tab_new, right=tab_old, left_on='image_id', right_on='image_id')

    tab = pd.read_csv(bbox_path)
    final_path = "C:/Users/Ismail/Documents/Projects/celeba_files/" + output_path
    create_folder(final_path)
    compteur = 0

    for k in range(tab.shape[0]):
        box = tab.iloc[k]
        image_id = box.image_id
        resize_img_path = '{}/{}'.format(final_path, image_id)
        img_path = global_path+'/'+image_id
        img = cv2.imread(img_path, 0)
        center_y = int(box.bbox_y + box.bbox_h//2 - box.bbox_h/16)
        center_x = int(box.bbox_x + box.bbox_w//2)
        if img.size > 0:

            crop_img = img[int(center_y-beta*box.bbox_h):int(center_y+beta*box.bbox_h),
                            int(center_x-alpha*box.bbox_w):int(center_x+alpha*box.bbox_w)]

            if crop_img.size>0: # ensure image is not empty
                resize_img = cv2.resize(crop_img, dsize=(36, 36))
                cv2.imwrite(resize_img_path, resize_img)
                compteur += 1
            else:
                tab_join = tab_join[tab_join.image_id != image_id]

        if compteur % 5000==0:
            print("{} images out of {} have been cropped".format(compteur, tab_join.shape[0]))

    tab_join.to_csv(new_attributes_path)
    print("All functional images have been cropped and put in the new directory")

if crop:
    if aligned:
        crop_well_dataset()
    elif not aligned:
        crop_dataset()

# Building datasets for each attribute
def build_dataset(attribute, positive_attribute, negative_attribute):

    final_path = "C:/Users/Ismail/Documents/Projects/celeba_files/"
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

if multi_folder:
    build_dataset('beard', 'No_Beard', 'Beard')
    build_dataset('eyeglasses', 'Eyeglasses', 'Naked_eye')
    build_dataset('hat', 'Wearing_Hat', 'No_Hat')
    build_dataset('gender', 'Male', 'Female')
    build_dataset('hairstyle', 'Bald', 'Hairy')
    build_dataset('mustache', 'Mustache', 'No_Mustache')