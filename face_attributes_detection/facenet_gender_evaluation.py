import pandas as pd
import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import platform
from sklearn.metrics import accuracy_score as acc

host = platform.node()
root_linux = "/dev/shm/data/celeba_files/"
root_windows = "C:/Users/Ismail/Documents/Projects/celeba_files/"
root_scaleway = '/root/data/celeba_files/'

if host == 'castor' or host == 'altair':  # Enrico's PCs
    root_path = root_linux
elif host == 'DESKTOP-AS5V6C3':  # Ismail's PC
    root_path = root_windows
elif host == 'scw-zealous-ramanujan' or host == 'scw-cranky-jang':
    root_path = root_scaleway
else:
    raise RuntimeError('Unknown host')

global_path = root_path
images_paths = global_path + "test_images/images/"
csv_path = global_path + "test_images/bigdb.csv"
model_filename = global_path + "facenet.h5"
shape, channel = 36, 1

def build_folder(path):
#build a directory and confirm execution with a message
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)

def globalize(path):
#make path in relation with an existing directory
    return global_path + path

def avg(liste):
#compute average of a list
    if type(liste)!=list:
        print("it is not a list")
    else:
        return sum(liste)/len(liste)

def labelize(label, flag = 'category'):
#transform the vector output of a final dense layer with softmax
#the most likely label gets one and the other takes 0
#as would .utils.to_categorical do to a binary categorical attributes
    if flag == 'category':
        if label == ' MALE':
            return 1
        elif label == ' FEMALE':
            return 0
    else:
        if label == 1:
            return ' MALE'
        elif label == 0:
            return ' FEMALE'

tab = pd.read_csv(csv_path)
tab.columns = ['filename', 'x', 'y', 'w', 'h', 'gender', 'age']

def build_dataset(alpha=0.5, beta=0.5, augmentation=False):
#augmentation boolean that does data augmentation with miroring effect if True
    images_set, labels = [], []
    #path_cropped = global_path+'cropped_test'
    #build_folder(path_cropped)

    for i in range(tab.shape[0]):
        path = images_paths+tab['filename'][i]
        label = tab['gender'][i]

        if label == ' MALE':
            num = 1
        elif label == ' FEMALE':
            num = 0
        else:
            continue

        x = tab['x'][i]
        y = tab['y'][i]
        w = tab['w'][i]
        h = tab['h'][i]

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) / 255
        center_y = int(y + h // 2 - h / 16)
        center_x = int(x + w // 2)
        im = img[int(center_y - beta * h):int(center_y + beta * h),
                   int(center_x - alpha * w):int(center_x + alpha * w)]

        if im.size>0:

            im = cv2.resize(im, (shape, shape))
            #cv2.imwrite(path_cropped+f'/img_{i}.jpg',im*255)
            images_set.append(im)
            labels.append(num)

            if augmentation:
                im_flip = cv2.flip(im, 1)
                images_set.append(im_flip)
                labels.append(num)

    return images_set, labels

def plot_confusion_matrix(cm, classes,filename,
                        normalize=False,
                        title='confusion matrix',
                        cmap='Blues'):
#plots and saves a confusion matrix with a given filename
#it is a heatmap inBlues to make things clearer
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)

alphas = list(np.arange(0.1,1,0.1))
betas = list(np.arange(0.1,1.2,0.1))

def model_analysis(model_file):
    model = tf.keras.models.load_model(model_file)
    confusion_path = globalize('confusion matrices')
    build_folder(confusion_path)
    for alpha in alphas:
        for beta in betas:
            if alpha < beta:
                file = open(confusion_path + "/accuracies.txt", "a")
                images, classes = build_dataset(alpha=alpha, beta=beta)

                images, classes = np.array(images), np.array(classes)
                images = images.reshape((images.shape[0], shape, shape, channel))

                # transform classes list to a binary matrix representation of the input so tensorflow can work with it
                classes = tf.keras.utils.to_categorical(classes, num_classes=2)

                predictions = model.predict(images)
                pred = np.argmax(predictions[0], axis=1)
                #d = {'gender_predicted':[labelize(element, flag='number') for element in pred]}
                labels = np.argmax(classes, axis=1)
                cm = confusion_matrix(y_true=labels , y_pred=pred)
                cm_plot_labels = ['Female', 'Male']
                plot_confusion_matrix(cm=cm, classes=cm_plot_labels,
                                      title="Confusion matrix",
                                      filename=confusion_path+f'/cm_alpha_{int(alpha*10)/10}_beta_{int(beta*10)/10}.jpg')
                file.write(f"\naccuracy for alpha={int(alpha*10)/10} and beta={int(beta*10)/10} is: {acc(labels, pred)}")

model_analysis(model_filename)