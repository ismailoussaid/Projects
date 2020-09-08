#!/usr/bin/env python3

import os
import sys
import time
import math
import getopt
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

import platform
host = platform.node()

root_linux = "/dev/shm/data/celeba_files/"
root_windows = "C:/Users/Ismail/Documents/Projects"
root_scaleway = '/home/deeplearn/data'
if host == 'castor' or host == 'altair':  # Enrico's PCs
    root_path = root_linux
elif host == 'ismailpcnamehere':  # Ismail's PC
    root_path = root_windows
elif host == 'scalewaynamehere':
    root_path = root_scaleway
else:
    raise RuntimeError('Unknown host')    

images_path = root_path + "cropped_images/"
global_path = root_path
output_path = root_path + "face_attributes_detection/"
attributes_path = root_path + "celeba_csv/list_attr_celeba.csv"



def build_folder(path):
    # build a directory and confirm execution with a message
    try:
        os.makedirs(path)
        print("<== {} directory created ==>")
    except OSError:
        print("Creation of the directory %s failed" % path)


def multiple_append(listes, elements):
    #append different elements to different lists in the same time
    if len(listes) != len(elements):
        #ensure, there is no mistake in the arguments
        raise RuntimeError("lists and elements do not have the same length")
    else:
        for l, e in zip(listes, elements):
            l.append(e)

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

## TODO: it's a pity to import keras (which is already in tensorflow) just to not re-implement this
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
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops


class FaceNet:
    def __init__(self, shape, channel, unit):
        self.shape = shape
        self.channel = channel
        self.unit = unit
        self.categs = ['gender', 'mustache', 'eyeglasses', 'beard', 'hat', 'bald']

    def build(self, size, num=2):
        # initialize the input shape and channel dimension (this code
        # assumes you are using TensorFlow which utilizes channels
        # last ordering)
        inputShape = (self.shape, self.shape, self.channel)

        # construct gender, mustache, bald, eyeglasses, beard & hat sub-networks
        inputs = Input(shape=inputShape)
        x = Conv2D(4, size, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(16, size, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        flat = Flatten()(x)
        net = Dense(self.unit)(flat)
        net = BatchNormalization(axis=-1)(net)
        net = Activation("relu")(net)

        outputs = []
        for c in self.categs:
            tmp_net = Dense(num)(net)
            tmp_net = Activation('softmax', name=c)(tmp_net)
            outputs.append(tmp_net)

        # create the model using our input (the batch of images) and
        # six separate outputs
        model = Model(inputs=inputs, outputs=outputs, name="facenet")

        # return the constructed network architecture
        return model


class CelebASequence(Sequence):
    TEST_PROPORTION = 0.1

    def __init__(self, attributes_path, images_path, batch_size, shape, channel, max_items=None):
        self.images_path = images_path
        self.batch_size = batch_size
        self.sizes = (shape, shape, channel)
        self.samples_per_data = 4
        if batch_size % self.samples_per_data != 0:  # VERY debatable
            raise NotImplementedError('Batch size has to be multiple of 4')

        self.attributes_tab = pd.read_csv(attributes_path)
        if max_items is not None:
            self.attributes_tab = self.attributes_tab.iloc[0:max_items]
        
        num_elements = len(self.attributes_tab['image_id'])
        
        print('Full Dataset has ' + str(num_elements) + ' elements (before augmentation)')
        indexes = np.arange(0, num_elements)
        self.input_train, self.input_test = train_test_split(indexes, test_size=self.TEST_PROPORTION, random_state=42)
        print(f'After split: {len(self.input_train)} train and {len(self.input_test)} test')
        self.set_mode_train()

        self.inner_batch_size = self.batch_size / self.samples_per_data
        
        self.attributes = ['Male', 'Mustache', 'Eyeglasses', 'No_Beard', 'Wearing_Hat', 'Bald']
        self.attr_mapper = {'Male':'gender' , 'Mustache': 'mustache', 'Eyeglasses': 'eyeglasses', 'No_Beard': 'beard', 'Wearing_Hat': 'hat', 'Bald': 'bald'}
        
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
        return math.floor((ln * self.samples_per_data) / self.batch_size)

    def __getitem__(self, idx):
        st, sp = int(idx * self.inner_batch_size), int((idx + 1) * self.inner_batch_size - 1)
        
        imgs = np.empty((self.batch_size, *self.sizes))
        atts = {'gender': [], 'mustache': [], 'eyeglasses': [], 'beard': [], 'hat': [], 'bald': []}

        j = 0
        for k in range(st, sp+1):
            if self.mode == 0:
                idx = self.input_train[k]
            else:
                idx = self.input_test[k]
        
            image_name = self.attributes_tab['image_id'][idx]
            
            img = cv2.cvtColor(cv2.imread(self.images_path + image_name), cv2.COLOR_BGR2GRAY)/255
            img = img.reshape(self.sizes)
            img_b = cv2.blur(img, (2,2)).reshape(self.sizes)
            img_m = cv2.flip(img, 1).reshape(self.sizes)
            img_mb = cv2.flip(img_b, 1).reshape(self.sizes)

            imgs[j,   :, :, :] = img
            imgs[j+1, :, :, :] = img_b
            imgs[j+2, :, :, :] = img_m
            imgs[j+3, :, :, :] = img_mb
            j += self.samples_per_data

            for a in self.attributes:
                name = self.attr_mapper[a]
                for b in range(0, self.samples_per_data):
                    atts[name].append(int((int(self.attributes_tab[a][idx]) + 1) / 2))
            
        out_attrs = {}
        for k, v in atts.items():
            out_attrs[k] = tf.keras.utils.to_categorical(v, num_classes=2)

        return (imgs, out_attrs)

    def get_results(self):  # unused method
        if self.mode != 1:
            raise RuntimeError('Not in test mode')
        atts = {'gender': [], 'mustache': [], 'eyeglasses': [], 'beard': [], 'hat': [], 'bald': []}
        for idx in self.input_test:
            for a in self.attributes:
                name = self.attr_mapper[a]
                for b in range(0, 4):
                    atts[name].append(int((int(self.attributes_tab[a][idx]) + 1) / 2))
        return atts


def main(epochs=10, max_items=None):
    batch_size = 64
    k_size = (3, 3)
    shape, channel, unit = 36, 1, 8
    
    seq = CelebASequence(attributes_path, images_path, batch_size, shape, channel, max_items)
    seq.set_mode_train()

    losses = {"gender": "categorical_crossentropy",
              "mustache": "categorical_crossentropy",
              "eyeglasses": "categorical_crossentropy" ,
              "beard": "categorical_crossentropy",
              "hat": "categorical_crossentropy",
              "bald": "categorical_crossentropy"}

    lossWeights = {"gender": 10, "mustache": 1, "eyeglasses": 5, "beard": 5, "hat": 1, "bald": 5}

    net = FaceNet(shape, channel, unit)
    model = net.build(k_size)
    model.summary()

    # initialize the optimizer and compile the model
    print("[INFO] compiling model...")
    opt = tf.optimizers.SGD(lr=0.001)
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=[f1, 'accuracy'])
    
    # fit labels with images with a ReduceLRonPlateau which divide learning by 10
    # if for 2 consecutive epochs, global validation loss doesn't decrease
    reducelronplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2, factor=0.1)
    model.fit(x=seq, batch_size=batch_size, epochs=epochs, callbacks=[reducelronplateau])

    seq.set_mode_test()
    evaluations = model.evaluate(seq, return_dict=True)
    for k, v in evaluations.items():
        print(k + ': ' + str(v))
    
    compute_flops = True
    if compute_flops:
        #to have the flops we have to do that with a h5 model
        #problem is that you can't compile model with a f1 measure as it doesn't exist in keras originally
        #so, i retrain the model in one epoch, save it and then, compute flops of the model
        folder = output_path
        build_folder(folder)
        model_filename = output_path + "facenet_flops_test.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_filename, monitor='val_loss', mode='min')
        mtl = model #FaceNet.build(size=k_size)

        # initialize the optimizer and compile the model
        print("[INFO] compiling flop model...")
        opt = tf.optimizers.SGD(lr=0.001)
        mtl.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])

        seq.set_mode_train()
        mtl.fit(x=seq, epochs=1, callbacks=[checkpoint])

        flop = get_flops(model_filename)

       
        with open(folder + "/perf.txt", "w+") as file:
            file.write(f"Here are the info for {k_size} kernel size model:" +'\n')
            if False:  # k-Fold stuff
                file.write(f"hat: {hat_avg*100}% +/- {hat_std*100}% (standard deviation)" +'\n')
                file.write(f"bald: {bald_avg*100}% +/- {bald_std*100}% (standard deviation)" +'\n')
                file.write(f"beard: {beard_avg*100}% +/- {beard_std*100}% (standard deviation)" +'\n')
                file.write(f"mustache: {mustache_avg*100}% +/- {mustache_std*100}% (standard deviation)" +'\n')
                file.write(f"gender: {gender_avg*100}% +/- {gender_std*100}% (standard deviation)" +'\n')
                file.write(f"eyeglasses: {eyeglasses_avg*100}% +/- {eyeglasses_std*100}% (standard deviation)" +'\n')
            file.write(f"Total flops are : {flop}")
    
    

def usage():
    print('./' + os.path.basename(__file__) + ' [options]')
    print('-e / --epochs N       Run training on N epochs [10]')
    print('-n / --num_items N    Use at most N items from the dataset [all]')
    sys.exit(-1)


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'e:n:', ['epochs=', 'num_items='])

    epochs = 10
    max_items = None
    for o, a in opts:
        if o in ('-e', '--epochs'):
            epochs = int(a)
        elif o in ('-n', '--num_items'):
            max_items = int(a)
        else:
            usage()

    main(epochs, max_items)
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

