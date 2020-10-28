import os
import sys
import getopt
from collections import defaultdict

import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow import optimizers as topt

from platform_settings import *
from common import *
from networks import FaceNet_NoGender
from sequences import *
from utils import *


k_sizes = [(3,3)]
first_convs = [4, 8]
second_convs = [8, 16]
units = [8, 16]
batch_sizes = [64, 128, 256]

def main(epochs, max_items, folds, skip):
    shape, channel, compute_flops = 36, 1, True
    losses = {"mustache": "categorical_crossentropy",
              "eyeglasses": "categorical_crossentropy",
              "beard": "categorical_crossentropy",
              "hat": "categorical_crossentropy",
              "bald": "categorical_crossentropy"}

    lossWeights = {"mustache": 1, "eyeglasses": 5, "beard": 5, "hat": 1, "bald": 5}

    dict_col = initialize_results(losses, compute_flops=compute_flops)

    # fit labels with images with a ReduceLRonPlateau which divide learning by 10
    # if for 2 consecutive epochs, global validation loss doesn't decrease
    reducelronplateau = ReduceLROnPlateau(monitor='loss', mode='min', patience=2, factor=0.1)
    earlystopping = EarlyStopping(monitor='loss', mode='min', patience=3)

    # to have the flops we have to do that with a h5 model
    # problem is that you can't compile model with a f1 measure as it doesn't exist in keras originally
    # so, I retrain the model in one epoch, save it and then, compute flops of the model
    if compute_flops:
        model_filename = root_path + "facenet_flops_test.h5"
        checkpoint = ModelCheckpoint(filepath=model_filename, monitor='loss', mode='min')

    opt = topt.SGD(lr=0.001)

    # build Sequence with data augmentation
    seq = CelebASequence(attributes_path, images_path, shape, channel, max_items=max_items)
    seq.augment(Mirroring())
    seq.augment(Blurring())
    seq.augment(Blurring(Mirroring()))
    seq.augment(MotionBlur('H'))
    seq.augment(MotionBlur('H', Mirroring()))

    s = 0
    for k_size in k_sizes:
        for batch_size in batch_sizes:
            # (Re)Initialize sequence on new batch size
            seq.prepare(batch_size)

            for first_conv in first_convs:
                for second_conv in second_convs:
                    if first_conv >= second_conv:
                        continue
                    for unit in units:
                        s+=1
                        print(f"Combinaison: {s} || {k_size} {batch_size} {first_conv} {second_conv} {unit}")

                        # Skip already-computed combinations
                        if s <= skip:
                            print('Skipped')
                            continue

                        #Creating the net for all these parameters
                        print("[INFO] compiling model...")
                        net = FaceNet_NoGender(shape, channel, unit, first_conv, second_conv)
                        model = net.build(k_size)
                        model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=[f1, 'accuracy'])

                        # Cross Validation 5-Fold
                        f1_scores = defaultdict(list)

                        for k in range(folds):
                            print(f'Fold {k}')
                            #Train
                            seq.set_mode_fold(k)
                            seq.set_mode_train()

                            model.fit(x=seq, epochs=epochs, callbacks=[reducelronplateau, earlystopping])

                            #Test
                            seq.set_mode_test()
                            evaluations = model.evaluate(seq)

                            # read evaluations by indexes found in model.metrics_names
                            for kk in losses.keys():
                                idx = model.metrics_names.index(kk + '_f1')
                                f1_scores[kk].append(evaluations[idx])

                        for key, value in losses.items():
                            score_array = np.array(f1_scores[key], dtype='float64')
                            average = np.mean(score_array)
                            stddev = np.std(score_array)
                            multiple_append([dict_col[key + " cv score"], dict_col[key + " std score"]], [average, stddev])

                        multiple_append([dict_col["number of Conv2D"], dict_col["number of Dense"], dict_col["kernel size"],
                                        dict_col["first conv"], dict_col["second conv"], dict_col["unit"], dict_col['batch_size']],
                                        [2, 1, k_size, first_conv, second_conv, unit, batch_size])

                        if compute_flops:
                            net_flops = FaceNet_NoGender(shape, channel, unit, first_conv, second_conv)
                            model_flops = net_flops.build(k_size)

                            seq_flops = CelebASequence(attributes_path, images_path, shape, channel, max_items=100)
                            seq_flops.prepare(batch_size)
                            opt_flops = topt.SGD(lr=0.001)

                            print("[INFO] compiling flops model...")
                            model_flops.compile(optimizer=opt_flops, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
                            model_flops.fit(x=seq_flops, epochs=1, callbacks=[checkpoint])

                            flops = get_flops(model_filename)
                            dict_col["flop"].append(flops)

                        print(dict_col)
                        df = pd.DataFrame(data=dict_col)
                        df.to_excel(root_path+"tab_comparison_no_gender_motion.xlsx")
                        print("Updated Dataframe is saved as xlsx")

    print("Final Dataframe is saved as xlsx")


def usage(max_items, epochs, folds, skip):
    print('./' + os.path.basename(__file__) + ' [options]')
    print(f'-e / --epochs N       Run training on N epochs [{epochs}]')
    print(f'-n / --num_items N    Use at most N items from the dataset [{"all" if max_items is None else str(max_items)}]')
    print(f'-f / --num_folds N    Uses 5-fold crossval, but runs only N fold(s) [{folds}]')
    print(f'-s / --skip N         Does not compute first N combinations [{skip}]')
    sys.exit(-1)


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'e:n:f:s:', ['epochs=', 'num_items=', 'num_folds=', 'skip='])

    max_items = None
    epochs = 10
    folds = 1
    skip = 0
    for o, a in opts:
        if o in ('-e', '--epochs'):
            epochs = int(a)
        elif o in ('-n', '--num_items'):
            max_items = int(a)
        elif o in ('-f', '--num_folds'):
            folds = int(a)
        elif o in ('-s', '--skip'):
            skip = int(a)
        else:
            usage(max_items, epochs, folds, skip)

    main(epochs, max_items, folds, skip)
    sys.exit(0)
