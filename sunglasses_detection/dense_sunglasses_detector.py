import tensorflow as tf
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd

epochs = 25
images_paths = "C:/Users/Ismail/Documents/QuividiData/cropped_images"
global_path = "C:/Users/Ismail/Documents/Projects"
TEST_PROPORTION = 0.05
shape, channel = 24, 1

def open_resize(path, liste):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)/255
    img = cv2.resize(img, (shape, shape)).reshape((shape, shape, channel))
    liste.append(img)

def build_folder(path, name):
    try:
        os.makedirs(path)
        print("<== {} directory created ==>".format(name))
    except OSError:
        print("Creation of the directory %s failed" % path)

def globalize(path):
    return global_path + path

def build_dataset(majority_amount): #maximum number of sunglasses examples in dataset
    compteur = 0
    image_set, labels = [], []

    for path in glob.glob(images_paths + '/*.jpg'):
        if 'no' in path:
            if compteur < majority_amount:
                open_resize(path, image_set)
                labels.append(0)
                compteur += 1
            else: continue
        else:
            open_resize(path, image_set)
            labels.append(1)

    return image_set, labels

def separate_test_train(input, target, TEST_PROPORTION):
    input, target = np.array(input), np.array(target)

    input_train, input_test, target_train, target_test = train_test_split(input, target, test_size=TEST_PROPORTION)

    print("images test has a shape of: {}".format(input_test.shape))
    print("classes test has a shape of: {}".format(target_test.shape))
    assert len(input_test) == len(target_test)

    print("images for training has a shape of: {}".format(input_train.shape))
    print("classes for training has a shape of: {}".format(target_train.shape))
    assert len(input_train) == len(target_train)

    return input_train, input_test, target_train, target_test

def labelize(outputs):
    for output in outputs:
        index_max = np.argmax(output)
        output[index_max] = 1
        index_min = np.argmin(output)
        output[index_min] = 0

images, classes = build_dataset(3000)
print("<== Dataset is loaded  ==>")

classes = tf.keras.utils.to_categorical(classes, num_classes=2)
images, images_test, classes, classes_test = separate_test_train(images, classes, 0.15)

batches = [256, 128, 64, 32, 16, 8, 4]
fs = [256, 128, 64, 32]
layers_number = 4

def build_model():

    path_model = globalize("/model_{}_layers/models").format(layers_number)
    path_logger = globalize("/model_{}_layers/csv_logger").format(layers_number)
    path_curves = globalize("/model_{}_layers/curves").format(layers_number)

    paths = [path_logger, path_curves, path_model]
    names = ['logger', 'loss curves', 'model']

    for k in range(len(paths)):
        path = paths[k]
        name = names[k]
        build_folder(path, name)

    estimation = [[1,0] for example in classes_test]
    true = classes_test
    score = f1_score(true, estimation, average='micro')
    print("f1-score for majority estimator : {}".format(score))
    scores, col_f, col_batch, layers = [score, 1-score], [0, 0], [0, 0], ['majority estimator', 'minority_estimator']

    for batch in batches:
        for f in fs:

            model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(shape, shape, channel)),
                                         tf.keras.layers.Dense(f, activation = tf.nn.relu),
                                         tf.keras.layers.Dense(f//2, activation=tf.nn.relu),
                                         tf.keras.layers.Dense(f//4, activation = tf.nn.relu),
                                         tf.keras.layers.Dense(2, activation = tf.nn.softmax)])

            #model.summary()

            # Compile the model
            model.compile(optimizer = 'adam', loss = "binary_crossentropy",
                          metrics = [tf.keras.metrics.Recall(name='recall'),
                                     tf.keras.metrics.Precision(name='precision')])

            # Callbacks
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path_model + "/model_f_{}_batch_{}.hdf5".format(f, batch),
                                                            save_best_only=True, period=epochs,
                                                            monitor='val_recall', mode='max')

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_recall', factor=0.2,
                                                             node='max', patience=3, min_lr=1e-06)

            csv_logger = tf.keras.callbacks.CSVLogger(path_logger+"/logger_model_f_{}_batch_{}.log".format(f, batch),
                                                      separator=",", append=False)

            history_dense = model.fit(images, classes, batch_size = batch, epochs = epochs,
                                      validation_split=0.1, verbose = 2, shuffle = True,
                                      callbacks=[checkpoint, reduce_lr, csv_logger])

            #print(history_dense.history.keys())
            print("Model trained")

            horizontal_axis = np.array([epoch for epoch in range(1, epochs+1)])

            # Plot the recall curves for training and validation
            training_recall = np.array(history_dense.history['recall'])
            validation_recall = np.array(history_dense.history['val_recall'])
            plt.plot(horizontal_axis, training_recall)
            plt.plot(horizontal_axis, validation_recall)
            plt.legend(['Training Recall', 'Validation Recall'])
            maximum = max(validation_recall)
            argmaximum = np.argmax(validation_recall)
            plt.title("maximum validation recall:{:.4f} for epoch: {}".format(maximum, horizontal_axis[argmaximum]))
            plt.savefig(path_curves+"/recall_model_f_{}_batch_{}.jpg".format(f, batch))
            plt.figure()

            # Plot the precision curves for training and validation
            training_precision = history_dense.history['precision']
            validation_precision = history_dense.history['val_precision']
            plt.plot(horizontal_axis, training_precision)
            plt.plot(horizontal_axis, validation_precision)
            plt.legend(['Training precision', 'Validation precision'])
            maximum = max(validation_precision)
            argmaximum = np.argmax(validation_precision)
            plt.title("maximum validation precision:{:.4f} for epoch: {}".format(maximum, horizontal_axis[argmaximum]))
            plt.savefig(path_curves+"/precision_model_f_{}_batch_{}.jpg".format(f, batch))
            plt.figure()

            plt.close('all')

            model = tf.keras.models.load_model(
                globalize("/model_{}_layers/models").format(layers_number) + "/model_f_{}_batch_{}.hdf5".format(f,
                                                                                                                batch))
            model.compile(optimizer='adam', loss="binary_crossentropy",
                          metrics=[tf.keras.metrics.Recall(name='recall'),
                                   tf.keras.metrics.Precision(name='precision')])

            print("Model loaded")

            predictions = model.predict(images_test)
            labelize(predictions)

            score = f1_score(true, predictions, average='micro')
            #print("f1 score for model {} layers and f = {} & batch = {} is: {}".format(layers_number, f, batch, score))
            col_f.append(f)
            col_batch.append(batch)
            scores.append(score)
            layers.append(layers_number)

    print("Building Dataframe")
    d = {'number of layers' : layers, 'f': col_f, 'batch size': col_batch, 'F1-score': scores}
    df = pd.DataFrame(data=d)
    df.to_excel(globalize("/model_{}_layers/").format(layers_number) + "resume_{}_layers.xlsx".format(layers_number))
    print("Dataframe is saved as xlsx")

build_model()