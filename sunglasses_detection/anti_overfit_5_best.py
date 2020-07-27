import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import statistics
from sklearn.model_selection import KFold

epochs = 8
images_paths = "C:/Users/Ismail/Documents/QuividiData/cropped_images"
global_path = "/"
TEST_PROPORTION = 0
shape, channel = 24, 1
layers_number = 21
conv_nb, dense_nb = 2, 1
n_split = 5
kf = KFold(n_splits=n_split,
           random_state=42)
AUGMENTATION = True
DROPOUT = True
EARLY_STOPPING = True
REGULARIZATION = False
BATCH_NORM = True

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
            if augmentation:
                open_flip_resize(path, image_set)
                labels.append(1)

    return image_set, labels

def labelize(outputs):
    #transform the vector output of a final dense layer with softmax
    #the most likely label gets one and the other takes 0
    #as would .utils.to_categorical do to a binary categorical attributes
    for output in outputs:
        index_max = np.argmax(output)
        output[index_max] = 1
        index_min = np.argmin(output)
        output[index_min] = 0

images, classes = build_dataset(3000, AUGMENTATION)
print("<== Dataset is loaded  ==>")

#transform classes list to a binary matrix representation of the input so tensorflow can work with it
classes = tf.keras.utils.to_categorical(classes, num_classes=2)

images, classes = np.array(images), np.array(classes)

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

def avg(liste):
    if type(liste)!=list:
        print("it is not a list")
    else:
        return sum(liste)/len(liste)

def save_excel(df, path):
    name = "/resume_"

    if DROPOUT == True:
        name += "dropout_"
    if EARLY_STOPPING ==  True:
        name+= "earlystopping_"
    if BATCH_NORM == True:
        name+= "batchnorm_"
    if REGULARIZATION == True:
        name+= "regularization_"

    name += "augmented_{}_layers_size_{}_{}.xlsx".format(layers_number, shape, shape)

    df.to_excel(path + name)

def architecture(nb_dense, param_dense, nb_conv, param_conv, d_rate=0.3, f_rate=0.5):
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
        if BATCH_NORM == True:
            layers = [tf.keras.layers.Conv2D(filters=param_conv[0], kernel_size=(3,3), input_shape=(shape, shape, channel)),
                  tf.keras.layers.MaxPool2D((2, 2)),
                      tf.keras.layers.BatchNormalization()]
        for i in range(1, nb_conv):
            layers += [tf.keras.layers.Conv2D(filters=param_conv[i],kernel_size=(3,3)),
                  tf.keras.layers.MaxPool2D((2, 2))]
            if DROPOUT == True:
                layers += [tf.keras.layers.Dropout(rate=d_rate)]

        layers.append(tf.keras.layers.Flatten())

        for j in range(nb_dense):
            layers.append(tf.keras.layers.Dense(units=param_dense[j], activation=tf.nn.relu))
            if DROPOUT:
                layers += [tf.keras.layers.Dropout(rate=f_rate)]

        layers.append(tf.keras.layers.Dense(units=2, activation=tf.nn.softmax))

        return tf.keras.Sequential(layers)

#Load the comparative analysis for the excel tab
# of the best architecture with Data Augmentation (mirroring): 2 Conv2D & 1 Dense Layer
path_excel = globalize("/best_models").format(layers_number, shape, shape) + "/resume_augmented_{}_layers_size_{}_{}.xlsx".format(layers_number, shape, shape)
tab = pd.read_excel(open(path_excel, 'rb'), index_col=0).drop(['number of Conv2D',
                                                               'number of Dense'], axis=1)
tab = tab.rename(columns={'F1-score':'f1'})
tab = tab.nlargest(5, 'f1')
N = tab.shape[0]
params = []

for k in range(N):
    model_param = tab.iloc[k].astype('int32')
    f = model_param['f']
    d = model_param['d']
    batch_size = model_param['batch size']
    params.append((f,d,batch_size))

dropout_d_rates, dropout_f_rates = np.arange(0.1, 1, 0.2), np.arange(0.1, 1, 0.2)

def build_model():
    #create folder to save all tha callbacks
    #build columns for the final resume
    #test all the combinaisons of batch size, conv2d's num_filter, dense's number of nodes

    path_model = globalize("/best_models").format(layers_number, shape, shape)
    path = path_model
    name = 'model'
    build_folder(path, name)

    majority_estimation = [[1,0] for example in classes]
    true = classes
    score = f1_score(true, majority_estimation, average='micro')
    print("f1-score for majority estimator : {}".format(score))
    scores, col_f, col_d, col_batch, conv, dense, execution_time, std_scores, d_rates, f_rates = [score, 1-score], [0, 0], \
                                                                                                 [0, 0], [0, 0], [0, 0],\
                                                                          ['majority estimator','minority_estimator'], \
                                                                          [0, 0], [0,0], [0,0], [0,0]

    for f,d,batch in params:
        for d_rate in dropout_d_rates:
            for f_rate in dropout_f_rates:
                model = architecture(dense_nb, [d], conv_nb, [f, f//2], d_rate=d_rate, f_rate=f_rate)

                # Compile the model
                model.compile(optimizer = 'adam', loss = "binary_crossentropy")
                print("Model compiled")

                # Callbacks
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                                 mode='min', patience=2, min_lr=1e-12)
                earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                                                 mode='min')
                i, score_list= 0, [] #put score to 0 to sum the new score and divide by 5

                for train_index, test_index in kf.split(classes):
                    i+=1
                    x_train = images[train_index]
                    y_train = classes[train_index]
                    x_test = images[test_index]
                    y_test = classes[test_index]

                    model.fit(x_train, y_train, batch_size = batch, epochs = epochs,
                              validation_split=0.1, verbose = 2, shuffle = True,
                              callbacks = [reduce_lr, earlystopping])
                    print("Model trained for the fold n.{}".format(i))

                    predictions = model.predict(x_test)

                    labelize(predictions) #gives 1 to the most likely label and 0 otherwise
                    print("Prediction done for the fold n.{}".format(i))

                    score_list.append(f1_score(y_test, predictions, average='micro')) #compute f1-score
                    print("Score computed for the fold n.{}".format(i))

                score_cv = avg(score_list)
                score_std = statistics.stdev(score_list)

                multiple_append([col_f, col_d, col_batch, scores, conv, dense, d_rates, f_rates, std_scores],
                                [f, d, batch, score_cv, conv_nb, dense_nb, d_rate, f_rate, score_std])
                print("Columns updated")

                dict_col = {'number of Conv2D': conv, 'number of Dense': dense, 'f': col_f, 'd': col_d,
                     'batch size': col_batch, 'dropout rate for dense': d_rates, 'dropout rate for conv': f_rates,
                     'F1-score': scores, 'F1-Score STD': std_scores}

                df = pd.DataFrame(data=dict_col)
                save_excel(df, path_model)
                print("Updated Dataframe is saved as xlsx")

    print("Final Dataframe is saved as xlsx")

build_model()