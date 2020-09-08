#!/usr/bin/env python3

import os
import sys
import time
import math
import statistics

import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Activation, Dense, Flatten, Input, BatchNormalization, Lambda
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score

import platform
host = platform.node()

root_linux = "/dev/shm/data/celeba_files/"
root_windows = "C:/Users/Ismail/Documents/Projects"
if host == 'castor':
    root_path = root_linux
elif host == 'scalewaynamehere':
    root_path = root_scaleway
elif host == 'ismailpcnamehere':
    root_path = root_windows
else:
    raise RuntimeError('Unknown host')    

images_path = root_path + "cropped_images/"
global_path = root_path
output_path = root_path + "face_attributes_detection/"
attributes_path = root_path + "celeba_csv/list_attr_celeba.csv"

EPOCHS = 10
k_size = (3,3)
shape, channel = 36, 1
TEST_PROPORTION = 0.1
unit = 8
n_split = 5
T= 5000
kf = KFold(n_splits=n_split,
           shuffle= True,
           random_state=42)

def build_folder(path):
    # build a directory and confirm execution with a message
    try:
        os.makedirs(path)
        print("<== {} directory created ==>")
    except OSError:
        print("Creation of the directory %s failed" % path)

def globalize(path):
    # make path in relation with an existing directory
    return global_path + path

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

def labelize(outputs):
    # transform the vector output of a final dense layer with softmax
    # the most likely label gets one and the other takes 0
    # as would .utils.to_categorical do to a binary categorical attributes
    return np.argmax(outputs, axis=1)

def pred_to_label(prediction, attribute):

    if attribute == 'gender':
        if prediction == 0:
            return 'female'
        else: return 'male'

    elif attribute == 'mustache':
        if prediction == 0:
            return 'no mustache'
        else: return 'mustache'

    elif attribute == 'eyeglasses':
        if prediction == 0:
            return 'no eyeglasses'
        else: return 'eyeglasses'

    elif attribute == 'beard':
        if prediction == 0:
            return 'no beard'
        else: return 'beard'

    elif attribute == 'hat':
        if prediction == 0:
            return 'no hat'
        else: return 'wearing hat'

    else:
        if prediction == 0:
            return 'hairy'
        else: return 'bald'

def predict(model, test_images, flag='class'):

    predictions, adapted_images = [], []
    predicted_gender, predicted_mustache, predicted_eyeglasses, \
    predicted_beard, predicted_hat, predicted_bald = [], [], [], [], [], []

    for image in test_images:
        img = image.reshape(-1, 36, 36, 1)
        prediction = model.predict(img)
        gender_predict, mustache_predict, eyeglasses_predict, \
        beard_predict, hat_predict, bald_predict = np.argmax(prediction, axis=2)
        if flag == 'class':
            multiple_append([predicted_gender, predicted_mustache, predicted_eyeglasses,
                         predicted_beard, predicted_hat, predicted_bald],
                        [gender_predict[0], mustache_predict[0], eyeglasses_predict[0],
                        beard_predict[0], hat_predict[0], bald_predict[0]])
        elif flag == 'label':
            multiple_append([predicted_gender, predicted_mustache, predicted_eyeglasses,
                             predicted_beard, predicted_hat, predicted_bald],
                            [pred_to_label(gender_predict[0], 'gender'),
                             pred_to_label(mustache_predict[0], 'mustache'),
                             pred_to_label(eyeglasses_predict[0], 'eyeglasses'),
                             pred_to_label(beard_predict[0], 'beard'),
                             pred_to_label(hat_predict[0], 'hat'),
                             pred_to_label(bald_predict[0], 'bald')])

    return predicted_gender, predicted_mustache, predicted_eyeglasses, \
           predicted_beard, predicted_hat, predicted_bald

def separate_test_train(inp, test_size=TEST_PROPORTION):
    #build training & testing set from target and images with a chosen proportion of testing
    input_train, input_test = train_test_split(inp, test_size=test_size, random_state=42)
    return input_train, input_test


def f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

def plot_epoch(history, metric, filename):
    #takes history of model.fit and extract training and validation metric data
    #save the curve in filename
    plt.figure()
    horizontal_axis = np.array([epoch for epoch in range(1, len(history.history[metric])+1)])
    # Plot the metric curves for training and validation
    training = np.array(history.history[metric])
    validation = np.array(history.history['val_{}'.format(metric)])
    plt.plot(horizontal_axis, training)
    plt.plot(horizontal_axis, validation)
    plt.legend(['Training {}'.format(metric), 'Validation {}'.format(metric)])
    if 'loss' not in metric:
        maximum = max(validation)
        argmaximum = np.argmax(validation)
        plt.title("max val {} :{:.4f} for epoch: {}".format(metric, maximum, horizontal_axis[argmaximum]))
    else:
        minimum = min(validation)
        argminimum = np.argmin(validation)
        plt.title("min val {} :{:.4f} for epoch: {}".format(metric, minimum, horizontal_axis[argminimum]))
    plt.savefig(filename)
    plt.close()

def get_flops(model_h5_path):
    # computes floating points operations for a h5 model
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops

class FaceNet:

    def build_branch(inputs, num=2, param=unit, size=k_size):

        x = inputs

        x = Conv2D(4, size, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(16, size, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        flat = Flatten()(x)
        net = Dense(param)(flat)
        net = BatchNormalization(axis=-1)(net)
        net = Activation("relu")(net)

        # define a branch of output layers for the number of different
        # gender categories

        gender_net = Dense(num)(net)
        gender_net = Activation("softmax", name="gender")(gender_net)

        mustache_net = Dense(num)(net)
        mustache_net = Activation("softmax", name="mustache")(mustache_net)

        bald_net = Dense(num)(net)
        bald_net = Activation("softmax", name="bald")(bald_net)

        eyeglasses_net = Dense(num)(net)
        eyeglasses_net = Activation("softmax", name="eyeglasses")(eyeglasses_net)

        beard_net = Dense(num)(net)
        beard_net = Activation("softmax", name="beard")(beard_net)

        hat_net = Dense(num)(net)
        hat_net = Activation("softmax", name="hat")(hat_net)

        return gender_net, mustache_net, eyeglasses_net, beard_net, hat_net, bald_net

    @staticmethod
    def build(width=shape, height=shape, channel=channel, param=unit, size=k_size):
        # initialize the input shape and channel dimension (this code
        # assumes you are using TensorFlow which utilizes channels
        # last ordering)
        inputShape = (height, width, channel)

        # construct gender, mustache, bald, eyeglasses, beard & hat sub-networks
        inputs = Input(shape=inputShape)
        genderBranch, mustacheBranch, eyeglassesBranch, \
        beardBranch, hatBranch, baldBranch = FaceNet.build_branch(inputs, param=param,
                                                                  size=size)

        # create the model using our input (the batch of images) and
        # six separate outputs
        model = Model(
            inputs=inputs,
            outputs=[genderBranch, mustacheBranch, eyeglassesBranch, beardBranch, hatBranch, baldBranch],
            name="facenet")

        # return the constructed network architecture
        return model

def adapt(x):
    return (np.array(list(x)*4)+1)/2

def anti_adapt(x):
    return (-np.array(list(x)*4)+1)/2

def perf(liste):
    return np.mean(liste), statistics.stdev(liste)



class CelebASequence(Sequence):
    def __init__(self, attributes_path, images_path, batch_size):
        self.images_path = images_path
        self.attributes_tab = pd.read_csv(attributes_path)
        self.batch_size = batch_size
        self.num_elements = len(self.attributes_tab['image_id'])
        
        print('Metadata has ' + str(self.num_elements) + ' elements (before augmentation)')
        
        indexes = np.arange(0, self.num_elements)
        self.input_train, self.input_test = train_test_split(indexes, test_size=TEST_PROPORTION, random_state=42)
        self.mode = 0 # train
        
        if batch_size % 4 != 0:  # very debatable
            raise NotImplementedError('Batch size has to be multiple of 4')
        
        self.inner_batch_size = self.batch_size / 4
        
        self.attributes = ['Male', 'Mustache', 'Bald', 'Eyeglasses', 'No_Beard', 'Wearing_Hat']
        
    def set_mode_train(self):
        self.mode = 0
        return
        
    def set_mode_test(self):
        self.mode = 1
        return
        
    def __len__(self):
        if self.mode == 0:
            ln = len(self.input_train)
        else:
            ln = len(self.input_test)
        return math.ceil(ln / self.inner_batch_size)

    def __getitem__(self, idx):
        st, sp = int(idx * self.inner_batch_size), int((idx + 1) * self.inner_batch_size - 1)
        
        images = []
        attributes = []
        
        for k in range(st, sp+1):
            if self.mode == 0:
                idx = self.input_train[k]
            else:
                idx = self.input_test[k]
        
            image_name = self.attributes_tab['image_id'][idx]
            image_attributes = []
            for a in self.attributes:
                image_attributes.append(self.attributes_tab[a][idx])
            
            img = cv2.cvtColor(cv2.imread(self.images_path + image_name), cv2.COLOR_BGR2GRAY)/255
            img = img.reshape((shape, shape, channel))
            img_b = cv2.blur(img, (2,2))
            img_m = cv2.flip(img, 1)
            img_mb = cv2.flip(img_b, 1)
            
            images.append(img)
            images.append(img_b)
            images.append(img_m)
            images.append(img_mb)
            
            attributes.append(image_attributes)
            attributes.append(image_attributes)
            attributes.append(image_attributes)
            attributes.append(image_attributes)
            #y={"gender": gender_train, "mustache": mustache_train, "eyeglasses": eyeglasses_train, "beard": beard_train, "hat": hat_train, "bald": bald_train},
            
        return images, attributes
        
        



def main():

    seq = CelebASequence(attributes_path, images_path, 64)
    seq.set_mode_train()


    losses = {"gender": "categorical_crossentropy",
              "mustache": "categorical_crossentropy",
              "eyeglasses": "categorical_crossentropy" ,
              "beard": "categorical_crossentropy",
              "hat": "categorical_crossentropy",
              "bald": "categorical_crossentropy"}

    lossWeights = {"gender": 10, "mustache": 1, "eyeglasses": 5, "beard": 5, "hat": 1, "bald": 5}

    model = FaceNet.build(size=k_size)
    model.summary()

    folder = output_path
    #build_folder(folder)

    # initialize the optimizer and compile the model
    print("[INFO] compiling model...")
    opt = tf.optimizers.SGD(lr=0.001)
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=[f1, 'accuracy'])

    reducelronplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min',
                                                             patience= 2, factor=0.1)

    # fit labels with images with a ReduceLRonPlateau which divide learning by 10
    # if for 2 consecutive epochs, global validation loss doesn't decrease

    hat_scores, bald_scores, beard_scores, \
    mustache_scores, gender_scores, eyeglasses_score = [], [], [], [], [], []
    
    model.fit(x=seq, epochs=EPOCHS, batch_size = 64, callbacks=[reducelronplateau])

    predictions = np.argmax(np.array(model.predict(x_val)), axis=2)

    gender_pred = predictions[0]
    mustache_pred = predictions[1]
    eyeglasses_pred = predictions[2]
    beard_pred = predictions[3]
    hat_pred = predictions[4]
    bald_pred = predictions[5]
    
    sys.exit(0)

    # Cross validation
    indices = kf.split(gender)

    for train_index, test_index in indices:
        i += 1
        x_train = images[train_index]
        gender_train = gender[train_index]
        hat_train = hat[train_index]
        mustache_train = mustache[train_index]
        bald_train = bald[train_index]
        beard_train = beard[train_index]
        eyeglasses_train = eyeglasses[train_index]

        x_val = images[test_index]
        gender_val = gender[test_index]
        hat_val = hat[test_index]
        mustache_val = mustache[test_index]
        bald_val = bald[test_index]
        beard_val = beard[test_index]
        eyeglasses_val = eyeglasses[test_index]

        model.fit(x=x_train,
                    y={"gender": gender_train, "mustache": mustache_train,
                        "eyeglasses": eyeglasses_train, "beard": beard_train,
                        "hat": hat_train, "bald": bald_train},
                    epochs=EPOCHS,
                    validation_split=0.1,
                    batch_size = 64,
                    callbacks=[reducelronplateau])

        predictions = np.argmax(np.array(model.predict(x_val)), axis=2)

        gender_pred = predictions[0]
        mustache_pred = predictions[1]
        eyeglasses_pred = predictions[2]
        beard_pred = predictions[3]
        hat_pred = predictions[4]
        bald_pred = predictions[5]

        acc_gender = accuracy_score(labelize(gender_val), gender_pred)
        f1_mustache = f1_score(labelize(mustache_val), mustache_pred, average='binary')
        f1_eyeglasses = f1_score(labelize(eyeglasses_val), eyeglasses_pred, average='binary')
        f1_beard = f1_score(labelize(beard_val), beard_pred, average='binary')
        f1_hat = f1_score(labelize(hat_val), hat_pred, average='binary')
        f1_bald = f1_score(labelize(bald_val), bald_pred, average='binary')

        multiple_append([hat_scores, bald_scores, beard_scores,
                         mustache_scores, gender_scores, eyeglasses_score],
                        [f1_hat, f1_bald, f1_beard, f1_mustache, acc_gender, f1_eyeglasses])


    hat_avg, hat_std = perf(hat_scores)
    bald_avg, bald_std = perf(bald_scores)
    beard_avg, beard_std = perf(beard_scores)
    mustache_avg, mustache_std = perf(mustache_scores)
    gender_avg, gender_std = perf(gender_scores)
    eyeglasses_avg, eyeglasses_std = perf(gender_scores)

    #to have the flops we have to do that with a h5 model
    #problem is that you can't compile model with a f1 measure as it doesn't exist in keras originally
    #so, i retrain the model in one epoch, save it and then, compute flops of the model
    model_filename = output_path + f"/facenet_flops_test.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath= model_filename,
                                                    monitor='val_loss', mode='min')
    mtl = FaceNet.build(size=k_size)

    # initialize the optimizer and compile the model
    print("[INFO] compiling flop model...")
    opt = tf.optimizers.SGD(lr=0.001)
    mtl.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])

    mtl.fit(x=images, y={"gender": gender, "mustache": mustache,
                        "eyeglasses": eyeglasses, "beard": beard,
                        "hat": hat, "bald": bald},
              epochs=1, validation_split=0.1, batch_size = 64, callbacks=[checkpoint])

    flop = get_flops(model_filename)

    file = open(folder + f"/perf.txt", "w+")

    file.write(f"Here are the info for {k_size} kernel size model:" +'\n')
    file.write(f"hat: {hat_avg*100}% +/- {hat_std*100}% (standard deviation)" +'\n')
    file.write(f"bald: {bald_avg*100}% +/- {bald_std*100}% (standard deviation)" +'\n')
    file.write(f"beard: {beard_avg*100}% +/- {beard_std*100}% (standard deviation)" +'\n')
    file.write(f"mustache: {mustache_avg*100}% +/- {mustache_std*100}% (standard deviation)" +'\n')
    file.write(f"gender: {gender_avg*100}% +/- {gender_std*100}% (standard deviation)" +'\n')
    file.write(f"eyeglasses: {eyeglasses_avg*100}% +/- {eyeglasses_std*100}% (standard deviation)" +'\n')
    file.write(f"Total flops are : {flop}")

    file.close()
    
    

if __name__ == '__main__':
    main()
    sys.exit(0)    
    
   

"""
#saves f1, accuracy, loss curves during training with respect to epoch (val, training)
print("[INFO] saving training curves...")

for key in losses.keys():
    plot_epoch(history_mtl, key+'_accuracy', folder + f"/mtl_acc_{key}.jpg")
    plot_epoch(history_mtl, key+'_f1', folder + f"/mtl_f1_{key}.jpg")
    plot_epoch(history_mtl, key+'_loss', folder + f"/mtl_loss_{key}.jpg")

plot_epoch(history_mtl, 'loss', output_path + f"/mtl_loss.jpg")
"""
