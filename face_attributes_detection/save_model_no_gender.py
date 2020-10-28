import os
import sys
import getopt

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow import optimizers as topt

from platform_settings import *
from common import *
from networks import FaceNet_NoGender
from sequences import *
from utils import *


def main(unit, first_conv, second_conv, batch_size, k_size, epochs, max_items):
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
    checkpoint = ModelCheckpoint(filepath=model_filename, monitor='loss', mode='min')

    opt = topt.SGD(lr=0.001)

    #Creating the net for all these parameters
    net = FaceNet_NoGender(shape, channel, unit, first_conv, second_conv)
    model = net.build(k_size)
    seq = CelebASequence(attributes_path, images_path, shape, channel, max_items=max_items)
    seq.augment(Mirroring())
    seq.augment(Blurring())
    seq.augment(Blurring(Mirroring()))
    seq.augment(MotionBlur('H'))
    seq.augment(MotionBlur('H', Mirroring()))
    seq.prepare(batch_size)

    # initialize the optimizer and compile the model
    print("[INFO] compiling flop model...")
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
    seq.set_mode_train()
    model.fit(x=seq, epochs=epochs, callbacks=[checkpoint])
    flop = get_flops(model_filename)
    file = open(root_path + "flop_final_model_bis.txt", "w+")
    file.write(f"model flop: {flop}")


def usage(epochs, max_items):
    print('./' + os.path.basename(__file__) + ' [options]')
    print(f'-e / --epochs N       Run training on N epochs [{epochs}]')
    print(f'-n / --num_items N    Use at most N items from the dataset [{"all" if max_items is None else str(max_items)}]')
    sys.exit(-1)


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'e:n:unit:first:second:batch:kernel:',
                               ["epochs =", "num_items =", 'unit=', 'first=', 'second=', 'batch_size=', "kernel_size="])

    unit, first_conv, second_conv, batch_size, k_size = 16, 16, 32, 144, (3, 3)
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
            usage(epochs, max_items)

    main(unit, first_conv, second_conv, batch_size, k_size, epochs, max_items)
    sys.exit(0)
