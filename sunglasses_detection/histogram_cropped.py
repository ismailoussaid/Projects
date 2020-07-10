# Libraries importation
import cv2
import glob
import matplotlib.pyplot as plt
import numpy
from numpy import median

# Histogram of sizes

def get_shapes(folder_images):

    i = 0
    m = len(glob.glob(folder_images + '/*.jpg'))
    widths, heights = [], []

    for path in glob.glob(folder_images + '/*.jpg'):

        image = cv2.imread(path)  # reading the image
        width, height = image.shape[0], image.shape[1]  # extracting height & width of the image
        widths.append(width)
        heights.append(height)
        i += 1

        if i % 500 == 0:
            print("the size of image n.{}/{} is measured".format(i, m))  # print every 50 images

    return widths, heights

"""
widths, heights = get_shapes("C:/Users/Ismail/Documents/QuividiData/cropped_images")

# Histograms of shapes
plt.hist(widths, bins=100)
w, ww, w_median = min(widths), int(sum(widths)/len(widths)), int(median(widths))
plt.title('minimum of widths is {}, average width is {} & median width is {}'.format(w, ww, w_median))
plt.savefig("C:/Users/Ismail/Documents/Projects/sunglasses_detection/cropped_width_histogram.jpg")
plt.figure()
plt.hist(heights, bins=100)
h, hh, h_median = min(heights), int(sum(heights)/len(heights)), int(median(heights))
plt.title('minimum of heights is {}, average height is {} & median height is {}'.format(h, hh, h_median))
plt.savefig("C:/Users/Ismail/Documents/Projects/sunglasses_detection/cropped_height_histogram.jpg")
"""

# Histogram of labels

def get_labels_info(folder_images):

    i = 0
    m = len(glob.glob(folder_images + '/*.jpg'))
    labels = []
    yes, no = 0, 0

    for path in glob.glob(folder_images + '/*.jpg'):

        i+=1

        if 'no' in path:
            labels.append('NO')
            no += 1

        else:
            labels.append('YES')
            yes += 1

        if i % 500 == 0:
            print("the label of image n.{}/{} is measured".format(i, m))  # print every 50 images

    return labels, (yes/m)*100, 100*(no/m)

labels, positive, negative = get_labels_info("C:/Users/Ismail/Documents/QuividiData/cropped_images")
plt.hist(labels)
plt.title("There are {:.1f}% sunglasses images and {:.1f}% no sunglasses images".format(positive, negative))
plt.savefig("C:/Users/Ismail/Documents/Projects/sunglasses_detection/labels_histogram.jpg")