import os
import sys
import getopt
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow import optimizers as topt

from platform_settings import *
from common import *
from networks import FaceNet_NoGender
from sequences import *
from utils import *

from contextlib import redirect_stdout

def plot_epoch(history, metric, filename):
    # takes history of model.fit and extract training and validation metric data
    # save the curve in filename
    plt.figure()
    horizontal_axis = np.array([epoch for epoch in range(1, len(history.history[metric]) + 1)])
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


unit, first_conv, second_conv, batch_size, k_size = 16, 4, 16, 64, (3,3)

def main(epochs, max_items):
    shape, channel, compute_flops = 36, 1, True
    losses = {"mustache": "categorical_crossentropy",
              "eyeglasses": "categorical_crossentropy",
              "beard": "categorical_crossentropy",
              "hat": "categorical_crossentropy",
              "bald": "categorical_crossentropy"}

    lossWeights = {"mustache": 1, "eyeglasses": 5, "beard": 5, "hat": 1, "bald": 5}

    # fit labels with images with a ReduceLRonPlateau which divide learning by 10
    # if for 2 consecutive epochs, global validation loss doesn't decrease
    reducelronplateau = ReduceLROnPlateau(monitor='loss', mode='min', patience=2, factor=0.1)
    earlystopping = EarlyStopping(monitor='loss', mode='min', patience=3)

    # to have the flops we have to do that with a h5 model
    # problem is that you can't compile model with a f1 measure as it doesn't exist in keras originally
    # so, i retrain the model in one epoch, save it and then, compute flops of the model
    model_filename = root_path + "facenet_flops_test.h5"
    checkpoint = ModelCheckpoint(filepath=model_filename, monitor='loss', mode='min')

    opt = topt.SGD(lr=0.001)

    seq = CelebASequence(attributes_path, images_path, shape, channel, max_items=max_items)
    seq.augment(Mirroring())
    seq.augment(Blurring())
    seq.augment(Blurring(Mirroring()))
    seq.augment(MotionBlur('H'))
    seq.augment(MotionBlur('H', Mirroring()))
    seq.prepare(batch_size)

    #Creating the net for all these parameters
    net = FaceNet_NoGender(shape, channel, unit, first_conv, second_conv)
    model = net.build(k_size)

    with open(root_path + 'modelsummary_no_gender.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()    

    # initialize the optimizer and compile the model
    print("[INFO] compiling model...")
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=[f1, 'accuracy'])

    #Cross Validation 5-Fold
    f1_scores = defaultdict(list)
    for k in range(5):
        #Train on 80%
        seq.set_mode_fold(k)
        seq.set_mode_train()

        model.fit(x=seq, epochs=epochs, callbacks=[reducelronplateau, earlystopping])

        #Test on 20%
        seq.set_mode_test()
        evaluations = model.evaluate(seq)
        print(f"evaluations are: {evaluations}")

        # read evaluations by indexes found in model.metrics_names
        for kk in losses.keys():
            idx = model.metrics_names.index(kk + '_f1')
            f1_scores[kk].append(evaluations[idx])

    scores = {}
    for kk in losses.keys():
        score_array = np.array(f1_scores[kk], dtype='float64')
        average = np.mean(score_array)
        stddev = np.std(score_array)
        scores[kk] = (average, stddev)

    ## Compute flop
    # new sequence
    seq_fold = CelebASequence(attributes_path, images_path, shape, channel, max_items=100)
    seq_fold.prepare(batch_size)
    # initialize the optimizer and compile the model
    print("[INFO] compiling flop model...")
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
    seq_fold.set_mode_fold(0)
    model.fit(x=seq_fold, epochs=1, callbacks=[checkpoint])
    flop = get_flops(model_filename)

    # writing new file with performance and flop
    file = open(root_path + "performance_cv_no_gender.txt", "w+")
    for key, value in losses.items():
        file.write(key + f" average score: {scores[key][0]}\n")
        file.write(key + f" std score: {scores[key][1]}\n")

    file.write(f"model flop: {flop}")


def usage(epochs, max_items):
    print('./' + os.path.basename(__file__) + ' [options]')
    print(f'-e / --epochs N       Run training on N epochs [{epochs}]')
    print(f'-n / --num_items N    Use at most N items from the dataset [{"all" if max_items is None else str(max_items)}]')
    sys.exit(-1)

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'e:n:', ['epochs=', 'num_items='])

    max_items = None
    epochs = 25
    for o, a in opts:
        if o in ('-e', '--epochs'):
            epochs = int(a)
        elif o in ('-n', '--num_items'):
            max_items = int(a)
        else:
            usage(epochs, max_items)

    main(epochs, max_items)
    sys.exit(0)