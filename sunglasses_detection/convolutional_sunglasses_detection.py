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
images_paths = "C:/Users/Ismail/Documents/QuividiData/cropped_images"
global_path = "C:/Users/Ismail/Documents/Projects"
TEST_PROPORTION = 0.05
shape, channel = 36, 1

def open_resize(path, liste):
    #resize an image loaded from path and adds it to a list such that it is ready for tensorflow (shape=(x,y,channel))
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)/255
    img = cv2.resize(img, (shape, shape)).reshape((shape, shape, channel))
    liste.append(img)

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

def build_dataset(majority_amount):
#majority_amount : maximum number of sunglasses examples in dataset
    compteur = 0
    image_set, labels = [], []

    for path in glob.glob(images_paths + '/*.jpg'):
        if 'no' in path:
            if compteur < majority_amount:
                #ensure that we have less than the authorized amount of majority class samples
                open_resize(path, image_set)
                labels.append(0)
                compteur += 1
            else: continue
        else:
            open_resize(path, image_set)
            labels.append(1)

    return image_set, labels

def separate_test_train(input, target, TEST_PROPORTION):
    #build training & testing set from target and images with a chosen proportion of testing
    input, target = np.array(input), np.array(target)

    input_train, input_test, target_train, target_test = train_test_split(input, target,
                                                                          test_size=TEST_PROPORTION,
                                                                          random_state=42)

    print("images test has a shape of: {}".format(input_test.shape))
    print("classes test has a shape of: {}".format(target_test.shape))
    assert len(input_test) == len(target_test)

    print("images for training has a shape of: {}".format(input_train.shape))
    print("classes for training has a shape of: {}".format(target_train.shape))
    assert len(input_train) == len(target_train)

    return input_train, input_test, target_train, target_test

def labelize(outputs):
    #transform the vector output of a final dense layer with softmax
    #the most likely label gets one and the other takes 0
    #as would .utils.to_categorical do to a binary categorical attributes
    for output in outputs:
        index_max = np.argmax(output)
        output[index_max] = 1
        index_min = np.argmin(output)
        output[index_min] = 0

images, classes = build_dataset(3000)
print("<== Dataset is loaded  ==>")

#transform classes list to a binary matrix representation of the input so tensorflow can work with it
classes = tf.keras.utils.to_categorical(classes, num_classes=2)

images, images_test, classes, classes_test = separate_test_train(images, classes, 0.15)

def plot_epoch(history, metric, filename):
    #takes history of model.fit and extract training and validation metric data
    #save the curve in filename
    horizontal_axis = np.array([epoch for epoch in range(1, epochs + 1)])
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
        print("listes and elements do not have the same length")
    else:
        for k in range(len(listes)):
            liste = listes[k]
            element = elements[k]
            liste.append(element)

def get_flops(model_h5_path):
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

def architecture(nb_dense, param_dense, nb_conv, param_conv):
    if nb_conv>len(param_conv):
        print("not enough conv parameters")
    elif nb_conv<len(param_conv):
        print("too much conv parameters")
    if nb_dense>len(param_dense):
        print("not enough dense parameters")
    else:
        print("too much dense parameters")

    if nb_conv==len(param_conv) and nb_dense==len(param_dense):
        layers = [tf.keras.layers.Conv2D(filters=param_conv[0], kernel_size=(3,3), input_shape=(shape, shape, channel)),
                  tf.keras.layers.MaxPool2D((2, 2))]

        for i in range(1, nb_conv):
            layers += [tf.keras.layers.Conv2D(filters=param_conv[i],kernel_size=(3,3)),
                  tf.keras.layers.MaxPool2D((2, 2))]

        layers.append(tf.keras.layers.Flatten())

        for j in range(nb_dense):
            layers.append(tf.keras.layers.Dense(units=param_dense[j], activation=tf.nn.relu))

        layers.append(tf.keras.layers.Dense(units=2, activation=tf.nn.softmax))

        return tf.keras.Sequential(layers)

batches = [256, 128, 64, 32, 16, 8, 4]
fs = [256, 128, 64, 32]
ds = [128, 64, 32, 16, 8, 4]
layers_number = 2_1
conv_nb, dense_nb = 2, 1

def build_model():
    #create folder to save all tha callbacks
    #build columns for the final resume
    #test all the combinaisons of batch size, conv2d's num_filter, dense's number of nodes

    path_model = globalize("/model_{}_layers/models").format(layers_number)
    path_logger = globalize("/model_{}_layers/csv_logger").format(layers_number)
    path_curves = globalize("/model_{}_layers/curves").format(layers_number)

    paths = [path_logger, path_curves, path_model]
    names = ['logger', 'loss curves', 'model']

    for k in range(len(paths)):
        path = paths[k]
        name = names[k]
        build_folder(path, name)

    majority_estimation = [[1,0] for example in classes_test]
    true = classes_test
    score = f1_score(true, majority_estimation, average='micro')
    print("f1-score for majority estimator : {}".format(score))
    scores, col_f, col_d, col_batch, conv, dense, execution_time, flops = [score, 1-score], [0, 0], [0, 0], [0, 0], [0, 0],\
                                                                          ['majority estimator','minority_estimator'], \
                                                                          [0, 0], [0, 0]

    for batch in batches:
        for f in fs:
            for d in ds:
                model = architecture(dense_nb, [d], conv_nb, [f, f//2])

                # Compile the model
                model.compile(optimizer = 'adam', loss = "binary_crossentropy",
                              metrics = [tf.keras.metrics.Recall(name='recall'),
                                         tf.keras.metrics.Precision(name='precision')])
                print("Model compiled")

                # Callbacks
                checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path_model+"/model_f_{}_d_{}_batch_{}.hdf5".format(f, d, batch),
                                                               save_best_only=True, period=epochs,
                                                               monitor='val_loss', mode='max')

                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                               node='max', patience=3, min_lr=1e-06)

                csv_logger = tf.keras.callbacks.CSVLogger(path_logger+"/logger_model_f_{}_d_{}_batch_{}.log".format(f, d, batch),
                                                          separator=",", append=False)

                history_conv = model.fit(images, classes, batch_size = batch, epochs = epochs,
                                          validation_split=0.1, verbose = 2, shuffle = True,
                                         callbacks = [checkpoint, reduce_lr, csv_logger])
                print("Model trained")

                metrics = ['recall', 'precision']
                filenames = [path_curves+"/recall_model_f_{}_d_{}_batch_{}.jpg".format(f, d, batch),
                            path_curves+"/precision_model_c_{}_d_{}_batch_{}.jpg".format(f, d, batch)]

                #plot recall & precision evolution per epoch during the training
                #also, the plot title is the max metric & the corresponding epoch
                for k in range(len(metrics)):
                    metric = metrics[k]
                    filename = filenames[k]
                    plot_epoch(history_conv, metric, filename)
                    plt.figure()

                plt.close('all')

                #compute flops of the  model
                flop = get_flops(path_model + "/model_f_{}_d_{}_batch_{}.hdf5".format(f, d, batch))

                # load best model (corresponding to the model for best epoch)
                model = tf.keras.models.load_model(path_model + "/model_f_{}_d_{}_batch_{}.hdf5".format(f, d, batch))
                model.compile(optimizer='adam', loss="binary_crossentropy",
                              metrics=[tf.keras.metrics.Recall(name='recall'),
                                       tf.keras.metrics.Precision(name='precision')])
                print("Model loaded")

                start = timeit.default_timer()
                predictions = model.predict(images_test)
                stop = timeit.default_timer()

                flop = get_flops(path_model + "/model_f_{}_d_{}_batch_{}.hdf5".format(f, d, batch))

                labelize(predictions) #gives 1 to the most likely label and 0 otherwise
                print("Prediction done")

                time = stop - start #compute execution time
                score = f1_score(true, predictions, average='micro') #compute f1-score
                print("Score computed")

                multiple_append([col_f, col_d, col_batch, scores, conv, dense, execution_time, flops],
                                [f, d, batch, score, conv_nb, dense_nb, time, flop])
                print("Columns updated")

    print("Building Dataframe")

    d = {'number of Conv2D' : conv, 'number of Dense' : dense, 'f': col_f, 'd': col_d,
         'batch size': col_batch, 'F1-score': scores, 'execution time': execution_time,
         'flops' : flops}
    df = pd.DataFrame(data=d)
    df.to_excel(globalize("/model_{}_layers/").format(layers_number) + "resume_{}_layers.xlsx".format(layers_number))

    print("Dataframe is saved as xlsx")

build_model()