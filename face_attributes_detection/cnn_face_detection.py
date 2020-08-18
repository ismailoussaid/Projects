import tensorflow as tf
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import timeit
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

epochs = 10
images_paths = "C:/Users/Ismail/Documents/Projects/celeba_files"
global_path = "C:/Users/Ismail/Documents/Projects"
shape, channel = 36, 1

def build_folder(path, name):
    #build a directory and confirm execution with a message
    try:
        os.makedirs(path)
        print("<== {} directory created ==>".format(name))
    except OSError:
        print("Creation of the directory %s failed" % path)

def globalize(path):
    #make path in relation with an existing directory
    return global_path + path

def labelize(outputs):
    #transform the vector output of a final dense layer with softmax
    #the most likely label gets one and the other takes 0
    #as would .utils.to_categorical do to a binary categorical attributes
    labels=[]

    for output in outputs:
        index_max = np.argmax(output)
        labels.append(index_max)

    return labels

def process(img):
    #resize an image loaded from path and adds it to a list such that it is ready for tensorflow (shape=(x,y,channel))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (shape, shape)).reshape((shape, shape, channel))
    return img

# Note that the validation data should not be augmented!
def flow_from_directory(classes, attribute, batch, size=(shape, shape)):

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            classes=classes,
            directory = images_paths+f'/{attribute}/train',
            target_size=size,
            batch_size=batch,
            class_mode='binary',
            color_mode="grayscale",
            shuffle=True,
            seed=42)

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    val_generator = val_datagen.flow_from_directory(
        classes=classes,
        directory=images_paths + f'/{attribute}/val',
        target_size=size,
        batch_size=batch,
        color_mode="grayscale",
        class_mode='binary',
        shuffle=True,
        seed=42)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        classes=classes,
        directory= images_paths + f'/{attribute}/test',
        target_size=size,
        color_mode="grayscale",
        batch_size=batch,
        class_mode='binary',
        shuffle=True,
        seed=42
    )

    return train_generator, val_generator, test_generator

def plot_epoch(history, metric, filename):
    #takes history of model.fit and extract training and validation metric data
    #save the curve in filename
    horizontal_axis = np.array([epoch for epoch in range(1, len(history.history[metric])+1)])
    # Plot the metric curves for training and validation
    training = np.array(history.history[metric])
    validation = np.array(history.history['val_{}'.format(metric)])
    plt.plot(horizontal_axis, training)
    plt.plot(horizontal_axis, validation)
    plt.legend(['Training {}'.format(metric), 'Validation {}'.format(metric)])
    maximum = max(validation)
    argmaximum = np.argmax(validation)
    plt.title("max val {} :{:.4f} for epoch: {}".format(metric, maximum, horizontal_axis[argmaximum]))
    plt.savefig(filename)

def multiple_append(listes, elements):
    #append different elements to different lists in the same time
    if len(listes) != len(elements):
        #ensure, there is no mistake in the arguments
        print("lists and elements do not have the same length")
    else:
        for k in range(len(listes)):
            liste = listes[k]
            element = elements[k]
            liste.append(element)

def architecture(nb_dense, param_dense, nb_conv, param_conv):
    if nb_conv>len(param_conv):
        print("not enough conv parameters")
    elif nb_conv<len(param_conv):
        print("too much conv parameters")
    if nb_dense>len(param_dense):
        print("not enough dense parameters")
    elif nb_dense<len(param_dense):
        print("too much dense parameters")

    if nb_conv==len(param_conv) and nb_dense==len(param_dense):
        layers = [tf.keras.layers.Conv2D(filters=param_conv[0], kernel_size=(3,3), input_shape=(shape, shape, channel))]

        for i in range(1, nb_conv):
            if i%2 == 1:
                layers += [tf.keras.layers.Conv2D(filters=param_conv[i],kernel_size=(3,3)),
                      tf.keras.layers.MaxPool2D((2, 2))]
            if i % 2 == 0:
                layers += [tf.keras.layers.Conv2D(filters=param_conv[i], kernel_size=(3, 3))]

        layers.append(tf.keras.layers.Flatten())

        for j in range(nb_dense):
            layers.append(tf.keras.layers.Dense(units=param_dense[j], activation=tf.nn.relu))

        layers.append(tf.keras.layers.Dense(units=2, activation=tf.nn.softmax))

        return tf.keras.Sequential(layers)

batches = [32]
fs = [64]
ds = [16]
conv_nb, dense_nb = 5, 1

def build_model():
    #create folder to save all tha callbacks
    #build columns for the final resume
    #test all the combinaisons of batch size, conv2d's num_filter, dense's number of nodes

    for batch in batches:

        train_gender, val_gender, test_gender = flow_from_directory(['male', 'female'], 'gender', batch)

        for f in fs:
            for d in ds:

                model = architecture(dense_nb, [d//2**k for k in range(dense_nb)], conv_nb, [f//2**k for k in range(conv_nb)])
                model.summary()
                model.compile(metrics=['accuracy'], loss='sparse_categorical_crossentropy',
                              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))

                # fits the model on batches with real-time data augmentation:
                history = model.fit_generator(train_gender, epochs=epochs, validation_data=val_gender)

                plot_epoch(history, 'accuracy', globalize('/face_attributes_detection/history_acc.jpg'))
                plot_epoch(history, 'loss', globalize('/face_attributes_detection/history_loss.jpg'))

build_model()