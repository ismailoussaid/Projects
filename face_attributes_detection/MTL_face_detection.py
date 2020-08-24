import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Activation, \
    Dense, Flatten, Input, BatchNormalization, Lambda
import pandas as pd
import time


images_paths = "C:/Users/Ismail/Documents/Projects/celeba_files"
global_path = "C:/Users/Ismail/Documents/Projects"
output_path = "C:/Users/Ismail/Documents/Projects/face_attributes_detection"
attributes_path = images_paths + "/celeba_csv/list_attr_celeba.csv"

EPOCHS = 50
k_size = (5,5)
shape, channel = 36, 1

def build_folder(path, name):
    # build a directory and confirm execution with a message
    try:
        os.makedirs(path)
        print("<== {} directory created ==>".format(name))
    except OSError:
        print("Creation of the directory %s failed" % path)

def globalize(path):
    # make path in relation with an existing directory
    return global_path + path

def labelize(outputs):
    # transform the vector output of a final dense layer with softmax
    # the most likely label gets one and the other takes 0
    # as would .utils.to_categorical do to a binary categorical attributes
    labels = []

    for output in outputs:
        index_max = np.argmax(output)
        labels.append(index_max)

    return labels

def open_process(path, liste):
    #resize an image loaded from path and adds it to a list such that it is ready for tensorflow (shape=(x,y,channel))
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)/255
    img = img.reshape((shape, shape, channel))
    liste.append(img)

def open_miror(path, liste):
    #resize an image loaded from path and adds it to a list such that it is ready for tensorflow (shape=(x,y,channel))
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)/255
    img = cv2.flip(img, 1).reshape((shape, shape, channel))
    liste.append(img)

def f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

def multiple_append(listes, elements):
    # append different elements to different lists in the same time
    if len(listes) != len(elements):
        # ensure, there is no mistake in the arguments
        print("lists and elements do not have the same length")
    else:
        for k in range(len(listes)):
            liste = listes[k]
            element = elements[k]
            liste.append(element)

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

            print(flops.total_float_ops)

class FaceNet:

    def build_branch(inputs, num=2):

        x = inputs

        x = Conv2D(4, k_size, padding="same", activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(16, k_size, padding="same", activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        flat = Flatten()(x)

        # define a branch of output layers for the number of different
        # gender categories
        gender_net = Dense(16, activation="relu")(flat)
        gender_net = BatchNormalization()(gender_net)
        gender_net = Dense(num)(gender_net)
        gender_net = Activation("softmax", name="gender")(gender_net)

        # define a branch of output layers for the number of different
        mustache_net = Dense(16, activation="relu")(flat)
        mustache_net = BatchNormalization()(mustache_net)
        mustache_net = Dense(num)(mustache_net)
        mustache_net = Activation("softmax", name="mustache")(mustache_net)

        bald_net = Dense(16, activation="relu")(flat)
        bald_net = BatchNormalization()(bald_net)
        bald_net = Dense(num)(bald_net)
        bald_net = Activation("softmax", name="bald")(bald_net)

        eyeglasses_net = Dense(16, activation="relu")(flat)
        eyeglasses_net = BatchNormalization()(eyeglasses_net)
        eyeglasses_net = Dense(num)(eyeglasses_net)
        eyeglasses_net = Activation("softmax", name="eyeglasses")(eyeglasses_net)

        beard_net = Dense(16, activation="relu")(flat)
        beard_net = BatchNormalization()(beard_net)
        beard_net = Dense(num)(beard_net)
        beard_net = Activation("softmax", name="beard")(beard_net)

        hat_net = Dense(16, activation="relu")(flat)
        hat_net = BatchNormalization()(hat_net)
        hat_net = Dense(num)(hat_net)
        hat_net = Activation("softmax", name="hat")(hat_net)

        return gender_net, mustache_net, eyeglasses_net, beard_net, hat_net, bald_net

    @staticmethod
    def build(width=shape, height=shape, channel=channel):
        # initialize the input shape and channel dimension (this code
        # assumes you are using TensorFlow which utilizes channels
        # last ordering)
        inputShape = (height, width, channel)

        # construct gender, mustache, bald, eyeglasses, beard & hat sub-networks
        inputs = Input(shape=inputShape)
        genderBranch, mustacheBranch, eyeglassesBranch, beardBranch, hatBranch, baldBranch = FaceNet.build_branch(inputs)

        # create the model using our input (the batch of images) and
        # six separate outputs
        model = Model(
            inputs=inputs,
            outputs=[genderBranch, mustacheBranch, eyeglassesBranch, beardBranch, hatBranch, baldBranch],
            name="facenet")

        # return the constructed network architecture
        return model

def adapt(x):
    return (np.array(list(x)*2)+1)/2

def anti_adapt(x):
    return (-np.array(list(x)*2)+1)/2

attributes_tab = pd.read_csv(attributes_path)
images = []
i=0

start = time.time()
print("[INFO] importing cropped images ...")

for path in glob.glob(images_paths+"/cropped_images/*.jpg"):
    i += 1
    open_process(path, images)

print("[INFO] horizontally flipping images ...")

for path in glob.glob(images_paths+"/cropped_images/*.jpg"):
    i += 1
    open_miror(path, images)

end = time.time()
print("importing 400k augmented images has taken " + str(end - start) + " seconds")

start = time.time()

print("[INFO] building attributes lists : 1 for minority class (Positive) and 0 for majority class...")

gender = adapt(attributes_tab['Male'])
mustache = adapt(attributes_tab['Mustache'])
bald = adapt(attributes_tab['Bald'])
eyeglasses = adapt(attributes_tab['Eyeglasses'])
beard = anti_adapt(attributes_tab['No_Beard'])
hat = adapt(attributes_tab['Wearing_Hat'])

end = time.time()
print("building labels list has taken " + str(end - start) + " seconds")

images, gender, mustache, eyeglasses, beard, hat, bald = np.array(images),\
                           np.array(tf.keras.utils.to_categorical(gender, num_classes=2)),\
                           np.array(tf.keras.utils.to_categorical(mustache, num_classes=2)),\
                            np.array(tf.keras.utils.to_categorical(eyeglasses, num_classes=2)),\
                            np.array(tf.keras.utils.to_categorical(beard, num_classes=2)),\
                            np.array(tf.keras.utils.to_categorical(hat, num_classes=2)), \
                            np.array(tf.keras.utils.to_categorical(bald, num_classes=2))


model = FaceNet.build()

losses = {"gender": "categorical_crossentropy",
          "mustache": "categorical_crossentropy",
          "eyeglasses": "categorical_crossentropy" ,
          "beard": "categorical_crossentropy",
          "hat": "categorical_crossentropy",
          "bald": "categorical_crossentropy"}

lossWeights = {"gender": 1, "mustache": 0.5, "eyeglasses": 1, "beard": 1, "hat": 0.5, "bald": 1}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = tf.optimizers.SGD(lr=0.001)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=[f1, 'accuracy'])

reducelronplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min',
                                                         patience= 2, factor=0.1)

# fit labels with images with a ReduceLRonPlateau which divide learning by 10
# if for 2 consecutive epochs, global validation loss doesn't decrease
history_mtl = model.fit(x=images, y={"gender": gender, "mustache": mustache,
                                     "eyeglasses": eyeglasses, "beard": beard,
                                     "hat": hat, "bald": bald},
          epochs=EPOCHS, validation_split=0.2, batch_size = 64, callbacks=[reducelronplateau])

#saves f1, accuracy, loss curves during training with respect to epoch (val, training)
print("[INFO] saving training curves...")

for key in losses.keys():
    plot_epoch(history_mtl, key+'_accuracy', output_path + f"/mtl_acc_{key}.jpg")
    plot_epoch(history_mtl, key+'_f1', output_path + f"/mtl_f1_{key}.jpg")
    plot_epoch(history_mtl, key+'_loss', output_path + f"/mtl_loss_{key}.jpg")

plot_epoch(history_mtl, 'loss', output_path + "/mtl_loss.jpg")

#rebuild new model to get flops
mtl = FaceNet.build()

opt = tf.optimizers.SGD(lr=0.001)
mtl.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])

#to have the flops we have to do that with a h5 model
#problem is that you can't compile model with a f1 measure as it doesn't exist in keras originally
#so, i retrain the model in one epoch, save it and then, compute flops of the model
model_filename = output_path + "/facenet_flops_test.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath= model_filename,
                                                monitor='val_loss', mode='min')

mtl.fit(x=images, y={"gender": gender, "mustache": mustache,
                    "eyeglasses": eyeglasses, "beard": beard,
                    "hat": hat, "bald": bald},
          epochs=1, validation_split=0.2, batch_size = 64, callbacks=[checkpoint])

get_flops(model_filename)