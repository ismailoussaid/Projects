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

def update(dict_col, att_dict, flag='initialize', compute_flops=True):
    if flag == 'initialize':
        dict_col["number of Conv2D"] = []
        dict_col["number of Dense"] = []
        dict_col["kernel size"] = []
        dict_col["first conv"] = []
        dict_col["second conv"] = []
        dict_col["unit"] = []
        dict_col["batch_size"] = []
        for key, value in att_dict.items():
            dict_col[key + " cv score"] = []
            dict_col[key + " std score"] = []
        if compute_flops == True:
            dict_col["flop"] = []

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

def pred_to_label(prediction, attribute):
    if attribute == 'gender':
        if prediction == 0:
            return 'female'
        else:
            return 'male'

    elif attribute == 'mustache':
        if prediction == 0:
            return 'no mustache'
        else:
            return 'mustache'

    elif attribute == 'eyeglasses':
        if prediction == 0:
            return 'no eyeglasses'
        else:
            return 'eyeglasses'

    elif attribute == 'beard':
        if prediction == 0:
            return 'no beard'
        else:
            return 'beard'

    elif attribute == 'hat':
        if prediction == 0:
            return 'no hat'
        else:
            return 'wearing hat'

    else:
        if prediction == 0:
            return 'hairy'
        else:
            return 'bald'

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

def avg(liste):
    if type(liste)!=list:
        print("it is not a list")
    else:
        return sum(liste)/len(liste)

def adapt(x):
    return (x+1)/2

def anti_adapt(x):
    return (-x+1)/2

class FaceNet:
    def __init__(self, shape, channel, unit, first_conv, second_conv):
        self.shape = shape
        self.channel = channel
        self.unit = unit
        self.first = first_conv
        self.second = second_conv
        self.categs = ['gender', 'mustache', 'eyeglasses', 'beard', 'hat', 'bald']

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

        kf = KFold(n_splits=n_split,
                   shuffle=True,
                   random_state=42)

        self.input_train, self.input_test = [], []
        for train_index, test_index in kf.split(indexes):
            multiple_append([self.input_train, self.input_test], [train_index, test_index])

        print(f'After 5-Fold split: {len(self.input_train)} train and {len(self.input_test)} test')
        self.set_mode_train()
        self.set_mode_fold(0)
        self.inner_batch_size = self.batch_size / self.samples_per_data

        self.attributes = ['Male', 'Mustache', 'Eyeglasses', 'No_Beard', 'Wearing_Hat', 'Bald']
        self.attr_mapper = {'Male': 'gender', 'Mustache': 'mustache', 'Eyeglasses': 'eyeglasses', 'No_Beard': 'beard',
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
        atts = {'gender': [], 'mustache': [], 'eyeglasses': [], 'beard': [], 'hat': [], 'bald': []}
        j = 0

        for k in range(st, sp):
            if self.mode == 0:
                index = self.input_train[self.fold][k]
            else:
                index = self.input_test[self.fold][k]

            image_name = self.attributes_tab['image_id'][index]
            img = cv2.cvtColor(cv2.imread(self.images_path + image_name), cv2.COLOR_BGR2GRAY) / 255
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
        atts = {'gender': [], 'mustache': [], 'eyeglasses': [], 'beard': [], 'hat': [], 'bald': []}
        for index in self.input_test:
            for a in self.attributes:
                name = self.attr_mapper[a]
                for b in range(0, 4):
                    if name != 'beard':
                        atts[name].append(adapt(self.attributes_tab[a][index]))
                    else:
                        atts[name].append(anti_adapt(self.attributes_tab[a][index]))

unit, first_conv, second_conv, batch_size, k_size = 16, 4, 16, 64, (3,3)

def main(epochs=25, max_items=None):
    shape, channel, compute_flops = 36, 1, True
    s = 0
    losses = {"gender": "categorical_crossentropy",
              "mustache": "categorical_crossentropy",
              "eyeglasses": "categorical_crossentropy",
              "beard": "categorical_crossentropy",
              "hat": "categorical_crossentropy",
              "bald": "categorical_crossentropy"}

    lossWeights = {"gender": 10, "mustache": 1, "eyeglasses": 5, "beard": 5, "hat": 1, "bald": 5}

    # fit labels with images with a ReduceLRonPlateau which divide learning by 10
    # if for 2 consecutive epochs, global validation loss doesn't decrease
    reducelronplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', mode='min', patience=2, factor=0.1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=3)

    # to have the flops we have to do that with a h5 model
    # problem is that you can't compile model with a f1 measure as it doesn't exist in keras originally
    # so, i retrain the model in one epoch, save it and then, compute flops of the model
    model_filename = root_path + "facenet_flops_test.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_filename, monitor='loss', mode='min')

    opt = tf.optimizers.SGD(lr=0.001)

    seq = CelebASequence(attributes_path, images_path, batch_size, shape, channel, max_items=max_items)
    s+=1
    print(f"combinaison: {s}")

    #Creating the net for all these parameters
    net = FaceNet(shape, channel, unit, first_conv, second_conv)
    model = net.build(k_size)

    with open(root_path + 'modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()    

    # initialize the optimizer and compile the model
    print("[INFO] compiling model...")
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=[f1, 'accuracy'])

    bald_list, beard_list, hat_list, mustache_list, gender_list, eyeglasses_list = [], [], [], [], [], []

    #Cross Validation 5-Fold
    for k in range(5):
        #Train on 80%
        seq.set_mode_fold(k)
        seq.set_mode_train()

        model.fit(x=seq, epochs=epochs, callbacks=[reducelronplateau, earlystopping])

        #Test on 20%
        seq.set_mode_test()
        evaluations = model.evaluate(seq)

        print(f"evaluations are: {evaluations}")

        bald_f1 = evaluations[-2]
        hat_f1 = evaluations[-4]
        beard_f1 = evaluations[-6]
        eyeglasses_f1 = evaluations[-8]
        mustache_f1 = evaluations[-10]
        gender_acc = evaluations [-11]

        multiple_append([bald_list, beard_list, hat_list, mustache_list, gender_list, eyeglasses_list],
                        [bald_f1, beard_f1, hat_f1, mustache_f1, gender_acc, eyeglasses_f1])

    attributes_list = [gender_list, mustache_list, eyeglasses_list, beard_list, hat_list, bald_list]
    score_cv, score_std = [], []

    for score_list in attributes_list:
        score_cv.append(avg(score_list))
        score_std.append(statistics.pstdev(np.array(score_list, dtype='float64')))

    #compute flop
    seq_fold = CelebASequence(attributes_path, images_path, batch_size, shape, channel,
                         max_items=100)

    # initialize the optimizer and compile the model
    print("[INFO] compiling flop model...")
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
    seq_fold.set_mode_fold(0)
    model.fit(x=seq, epochs=1, callbacks=[checkpoint])

    #writing new file with performance and flop
    flop = get_flops(model_filename)

    file = open(root_path + "performance_cv.txt", "w+")
    i = 0
    for key, value in losses.items():
        file.write(key + f" average score: {score_cv[i]}\n")
        file.write(key + f" std score: {score_std[i]}\n")
        i += 1
    file.write(f"model flop: {flop}")

def usage():
    print('./' + os.path.basename(__file__) + ' [options]')
    print('-e / --epochs N       Run training on N epochs [10]')
    print('-n / --num_items N    Use at most N items from the dataset [all]')
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
            usage()

    main(epochs, max_items)
    sys.exit(0)