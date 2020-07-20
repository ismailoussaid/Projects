import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import timeit
from sklearn.model_selection import KFold

epochs = 10
images_paths = "C:/Users/Ismail/Documents/QuividiData/cropped_images"
global_path = "C:/Users/Ismail/Documents/Projects"
TEST_PROPORTION = 0.05
shape, channel = 24, 1
layers_number = 21
conv_nb, dense_nb = 2, 1
n_split = 5
kf = KFold(n_splits=n_split, random_state=42)

def open_resize(path, liste):
    #resize an image loaded from path and adds it to a list such that it is ready for tensorflow (shape=(x,y,channel))
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)/255
    img = cv2.resize(img, (shape, shape))
    liste.append(img.reshape(shape, shape, channel))

def open_flip_resize(path, liste):
    #resize an image loaded from path and adds it to a list such that it is ready for tensorflow (shape=(x,y,channel))
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)/255
    img = cv2.resize(img, (shape, shape))
    img = cv2.flip(img, 1)
    liste.append(img.reshape(shape, shape, channel))

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

def build_dataset(majority_amount, augmentation):
#majority_amount : maximum number of sunglasses examples in dataset
#augmentation boolean that does data augmentation with miroring effect on minority images if True
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
            if augmentation == True:
                open_flip_resize(path, image_set)
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

images, classes = build_dataset(3000, False)
print("<== Dataset is loaded  ==>")

#transform classes list to a binary matrix representation of the input so tensorflow can work with it
classes = tf.keras.utils.to_categorical(classes, num_classes=2)

images, images_test, classes, classes_test = separate_test_train(images, classes, 0.15)

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

def architecture(nb_dense, param_dense, nb_conv, param_conv):
    if nb_conv>len(param_conv):
        print("not enough conv parameters")
    elif nb_conv<len(param_conv):
        print("too much conv parameters")
    if nb_dense>len(param_dense):
        print("not enough dense parameters")
    elif nb_dense<len(param_dense):
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

#Load the comparative analysis for the excel tab of the best architecture: 2 Conv2D & 1 Dense Layer
path_excel = globalize("/model_{}_layers_size_{}_{}/resume_{}_layers_size_24_24.xlsx").format(layers_number, shape, shape, layers_number)
tab = pd.read_excel(open(path_excel, 'rb'), index_col=0).drop(['score/time', 'number of Conv2D',
                                                               'number of Dense', 'execution time'], axis=1)
tab = tab.rename(columns={'f: number of filter for the first convolutional layer':'f',
                          'F1-score on test':'f1',
                          'd: number of node for the dense layer':'d'})
tab = tab[tab.f1>0.98]
N = tab.shape[0]
params = []

for k in range(N):
    model_param = tab.iloc[k].astype('int32')
    f = model_param['f']
    d = model_param['d']
    batch_size = model_param['batch size']
    params.append((f,d,batch_size))

def build_model():
    #create folder to save all tha callbacks
    #build columns for the final resume
    #test all the combinaisons of batch size, conv2d's num_filter, dense's number of nodes

    path_model = globalize("/best_models").format(layers_number, shape, shape)
    path = path_model
    name = 'model'
    build_folder(path, name)

    majority_estimation = [[1,0] for example in classes_test]
    true = classes_test
    score = f1_score(true, majority_estimation, average='micro')
    print("f1-score for majority estimator : {}".format(score))
    scores, col_f, col_d, col_batch, conv, dense, execution_time = [score, 1-score], [0, 0], [0, 0], [0, 0], [0, 0],\
                                                                          ['majority estimator','minority_estimator'], \
                                                                          [0, 0]

    for f,d,batch in params:

        model = architecture(dense_nb, [d], conv_nb, [f, f//2])

        # Compile the model
        model.compile(optimizer = 'adam', loss = "binary_crossentropy")
        print("Model compiled")

        # Callbacks
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                         node='max', patience=1, min_lr=1e-12)
        i, score = 0, 0 #put score to 0 to sum the new score and divide by 5

        for train_index, test_index in kf.split(classes):
            i+=1
            x_train = images[train_index]
            y_train = classes[train_index]
            x_test = images[test_index]
            y_test = classes[test_index]

            model.fit(x_train, y_train, batch_size = batch, epochs = epochs,
                      validation_split=0.1, verbose = 2, shuffle = True,
                      callbacks = [reduce_lr])
            print("Model trained for the fold n.{}".format(i))

            predictions = model.predict(x_test)

            labelize(predictions) #gives 1 to the most likely label and 0 otherwise
            print("Prediction done for the fold n.{}".format(i))

            score += f1_score(y_test, predictions, average='micro')/5 #compute f1-score
            print("Score computed for the fold n.{}".format(i))

        multiple_append([col_f, col_d, col_batch, scores, conv, dense],
                        [f, d, batch, score, conv_nb, dense_nb])
        print("Columns updated")

        d = {'number of Conv2D': conv, 'number of Dense': dense, 'f': col_f, 'd': col_d,
             'batch size': col_batch, 'F1-score': scores}

        df = pd.DataFrame(data=d)
        df.to_excel(path_model + "/resume_{}_layers_size_{}_{}.xlsx".format(layers_number, shape, shape))

        print("Updated Dataframe is saved as xlsx")

    print("Final Dataframe is saved as xlsx")

build_model()