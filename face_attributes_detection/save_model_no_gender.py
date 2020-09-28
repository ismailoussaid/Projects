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
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Activation, Dense, Flatten, Input, \
    BatchNormalization, Lambda
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import KFold
import platform
from contextlib import redirect_stdout

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

global_path = root_path

if aligned:
    images_path = root_path + "well_cropped_images/"
    attributes_path = root_path + "celeba_csv/list_attr_celeba_aligned.csv"
else:
    images_path = root_path + "cropped_images/"
    attributes_path = root_path + "celeba_csv/list_attr_celeba.csv"

def build_folder(path):
    # build a directory and confirm execution with a message
    try:
        os.makedirs(path)
        print("<== {} directory created ==>")
    except OSError:
        print("Creation of the directory %s failed" % path)

def multiple_append(listes, elements):
    # append different elements to different lists in the same time
    if len(listes) != len(elements):
        # ensure, there is no mistake in the arguments
        raise RuntimeError("lists and elements do not have the same length")
    else:
        for l, e in zip(listes, elements):
            l.append(e)

def labelize(outputs):
    # transform the vector output of a final dense layer with softmax
    # the most likely label gets one and the other takes 0
    # as would .utils.to_categorical do to a binary categorical attributes
    return np.argmax(outputs, axis=1)

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

def adapt(x):
    return (x + 1) / 2

def anti_adapt(x):
    return (-x + 1) / 2

class FaceNet:
    def __init__(self, shape, channel, unit, first_conv, second_conv):
        self.shape = shape
        self.channel = channel
        self.unit = unit
        self.first = first_conv
        self.second = second_conv
        self.categs = ['mustache', 'eyeglasses', 'beard', 'hat', 'bald']

    def build(self, size, num=2):
        # initialize the input shape and channel dimension (this code
        # assumes you are using TensorFlow which utilizes channels
        # last ordering)
        inputShape = (self.shape, self.shape, self.channel)

        # construct gender, mustache, bald, eyeglasses, beard & hat sub-networks
        inputs = Input(shape=inputShape)
        x = Conv2D(self.first, size, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(self.second, size, padding="same")(x)
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

    def __init__(self, attributes_path, images_path, batch_size, shape, channel, max_items=None, n_split=5):
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

        kf = KFold(n_splits=n_split, shuffle=True)

        self.input_train, self.input_test = [], []
        for train_index, test_index in kf.split(indexes):
            multiple_append([self.input_train, self.input_test], [train_index, test_index])

        print(f'After 5-Fold split: {len(self.input_train)} train and {len(self.input_test)} test')
        self.set_mode_train()
        self.set_mode_fold(0)
        self.inner_batch_size = self.batch_size / self.samples_per_data

        self.attributes = ['Mustache', 'Eyeglasses', 'No_Beard', 'Wearing_Hat', 'Bald']
        self.attr_mapper = {'Mustache': 'mustache', 'Eyeglasses': 'eyeglasses', 'No_Beard': 'beard',
                            'Wearing_Hat': 'hat', 'Bald': 'bald'}

    def set_mode_train(self):
        self.mode = 0

    def set_mode_test(self):
        self.mode = 1

    def set_mode_fold(self, num_fold):
        self.fold = num_fold

    def __len__(self):
        if self.mode == 0:
            ln = len(self.input_train[self.fold])
        else:
            ln = len(self.input_test[self.fold])
        return math.floor((ln * self.samples_per_data) / self.batch_size)

    def __getitem__(self, idx):
        st, sp = int(idx * self.inner_batch_size), int((idx + 1) * self.inner_batch_size)

        imgs = np.empty((self.batch_size, *self.sizes))
        atts = {'mustache': [], 'eyeglasses': [], 'beard': [], 'hat': [], 'bald': []}
        j = 0

        for k in range(st, sp):
            if self.mode == 0:
                index = self.input_train[self.fold][k]
            else:
                index = self.input_test[self.fold][k]

            image_name = self.attributes_tab['image_id'][index]
            im = cv2.imread(self.images_path + image_name)
            img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
            img = img.reshape(self.sizes)
            img_b = cv2.blur(img, (2, 2)).reshape(self.sizes)
            img_m = cv2.flip(img, 1).reshape(self.sizes)
            img_mb = cv2.flip(img_b, 1).reshape(self.sizes)

            imgs[j, :, :, :] = img
            imgs[j + 1, :, :, :] = img_b
            imgs[j + 2, :, :, :] = img_m
            imgs[j + 3, :, :, :] = img_mb
            j += self.samples_per_data

            for a in self.attributes:
                name = self.attr_mapper[a]
                for b in range(0, self.samples_per_data):
                    if name != 'beard':
                        atts[name].append(adapt(self.attributes_tab[a][index]))
                    else:
                        atts[name].append(anti_adapt(self.attributes_tab[a][index]))

        out_attrs = {}
        for k, v in atts.items():
            out_attrs[k] = tf.keras.utils.to_categorical(v, num_classes=2)

        return (imgs, out_attrs)

    def get_results(self):  # unused method
        if self.mode != 1:
            raise RuntimeError('Not in test mode')
        atts = {'mustache': [], 'eyeglasses': [], 'beard': [], 'hat': [], 'bald': []}
        for index in self.input_test:
            for a in self.attributes:
                name = self.attr_mapper[a]
                for b in range(0, 4):
                    if name != 'beard':
                        atts[name].append(adapt(self.attributes_tab[a][index]))
                    else:
                        atts[name].append(anti_adapt(self.attributes_tab[a][index]))

def main(unit, first_conv, second_conv, batch_size, k_size, epochs=25, max_items=None):
    shape, channel, compute_flops = 36, 1, True
    losses = {"mustache": "categorical_crossentropy",
              "eyeglasses": "categorical_crossentropy",
              "beard": "categorical_crossentropy",
              "hat": "categorical_crossentropy",
              "bald": "categorical_crossentropy"}

    lossWeights = {"mustache": 1, "eyeglasses": 5, "beard": 5, "hat": 1, "bald": 5}

    # to have the flops we have to do that with a h5 model
    # problem is that you can't compile model with a f1 measure as it doesn't exist in keras originally
    # so, i retrain the model in one epoch, save it and then, compute flops of the model
    model_filename = root_path + "facenet_bis.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_filename, monitor='loss', mode='min')

    opt = tf.optimizers.SGD(lr=0.001)

    #Creating the net for all these parameters
    net = FaceNet(shape, channel, unit, first_conv, second_conv)
    model = net.build(k_size)
    seq = CelebASequence(attributes_path, images_path, batch_size, shape, channel, max_items=max_items)

    # initialize the optimizer and compile the model
    print("[INFO] compiling flop model...")
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
    seq.set_mode_train()
    model.fit(x=seq, epochs=epochs, callbacks=[checkpoint])
    flop = get_flops(model_filename)
    file = open(root_path + "flop_final_model_bis.txt", "w+")
    file.write(f"model flop: {flop}")

def usage():
    print('./' + os.path.basename(__file__) + ' [options]')
    print('-e / --epochs N       Run training on N epochs [10]')
    print('-n / --num_items N    Use at most N items from the dataset [all]')
    sys.exit(-1)

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'e:n:unit:first:second:batch:kernel:',
                               ["epochs =", "num_items =", 'unit=', 'first=', 'second=', 'batch_size=', "kernel_size="])

    unit, first_conv, second_conv, batch_size, k_size = 16, 4, 16, 64, (3, 3)
    max_items = None
    epochs = 50

    for o, a in opts:
        if o in ('-e', '--epochs'):
            epochs = int(a)
        elif o in ('-n', '--num_items'):
            max_items = int(a)
        elif o in ('-unit', '--unit'):
            unit = int(a)
        elif o in ('-first', '--first'):
            first_conv = int(a)
        elif o in ('-second', '--second'):
            second_conv = int(a)
        elif o in ('-batch', '--batch_size'):
            batch_size = int(a)
        elif o in ('-kernel', '--kernel_size'):
            k_size = int(a)
        else:
            usage()

    main(unit, first_conv, second_conv, batch_size, k_size, epochs, max_items)
    sys.exit(0)