import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import os
import cv2
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

epochs = 15
images_paths = "C:/Users/Ismail/Documents/QuividiData/cropped_images"
global_path = "/"
shape, channel = 24, 1
layers_number = 21
conv_nb, dense_nb = 2, 1
AUGMENTATION = True
DROPOUT = False
EARLY_STOPPING = True
REGULARIZATION = False
BATCH_NORM = False
BLURR = False
TEST_PROPORTION = 0.2

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

def avg(liste):
#compute average of a list
    if type(liste)!=list:
        print("it is not a list")
    else:
        return sum(liste)/len(liste)

#Load the comparative analysis for the excel tab of the best architecture: 2 Conv2D & 1 Dense Layer
filename_basic =  "/resume_augmented_21_layers_size_24_24.xlsx"
filename_dropout_earlystopping = "/resume_dropout_earlystopping_augmented_21_layers_size_24_24.xlsx"
filename_blurr_dropout_earlystopping_batchnorm = "/resume_blurr_dropout_earlystopping_batchnorm_augmented_21_layers_size_24_24.xlsx"
filename_blurr_dropout_earlystopping_regularization = "/resume_blurr_dropout_earlystopping_regularization_augmented_21_layers_size_24_24.xlsx"
filename_dropout_earlystopping_batchnorm = "/resume_dropout_earlystopping_batchnorm_augmented_21_layers_size_24_24.xlsx"
filename_dropout_earlystopping_regularization = "/resume_blurr_dropout_earlystopping_regularization_augmented_21_layers_size_24_24.xlsx"
filename_blurr_dropout_earlystopping_batchnorm_regularization = "/resume_blurr_dropout_earlystopping_batchnorm_regularization_augmented_21_layers_size_24_24.xlsx"
filename_blurr_dropout_earlystopping = "/resume_blurr_dropout_earlystopping_augmented_21_layers_size_24_24.xlsx"

filenames = [filename_dropout_earlystopping_batchnorm, filename_dropout_earlystopping_regularization,
            filename_basic, filename_dropout_earlystopping, filename_blurr_dropout_earlystopping,
            filename_blurr_dropout_earlystopping_batchnorm_regularization,
            filename_blurr_dropout_earlystopping_batchnorm, filename_blurr_dropout_earlystopping_regularization]

def analysis(filename):
#takes excel file of F1 score of each trained model
#prints the average score over all the models, the 20, 10 and 5 best model and its standard deviation
#explains in which case we are (batchnorm, etc.)
    path_excel = globalize("/best_models").format(layers_number, shape, shape) + filename
    tab = pd.read_excel(open(path_excel, 'rb'), index_col=0).drop(['number of Conv2D',
                                                                   'number of Dense'], axis=1)
    tab = tab.iloc[2:]
    tab = tab.rename(columns={'F1-score':'f1', 'F1-Score STD':'std'})
    M =  tab.shape[0]
    Ns = [tab.shape[0], 20, 10, 5]

    for N in Ns:
        tab = tab.nlargest(N, 'f1')
        std_mean = round(tab['std'].mean() * 100, 2)
        score_mean = round(tab.f1.mean() * 100, 2)
        tag = ''
        if 'blurr' in filename:
            tag += 'Blurred images'
        if 'batchnorm' in filename:
            if tag != '':
                tag += ', Batch Normalization'
            else:
                tag += 'Batch Normalization'
        if 'regularization' in filename:
            if tag != '':
                tag += ', Regularization'
            else:
                tag += 'Regularization'
        if N != M:
            print(f" For the {N} best models with " + tag + f", the average F1 is: {score_mean}%"
                      f" and average STD is: {std_mean}%")
        else:
            print(f" For all the models with " + tag + f", the average F1 is: {score_mean}%"
                                                            f" and average STD is: {std_mean}%")

    print("Best model in this case had F1: {}%".format(round(max(tab['f1'])*100,2)))

#analysis(filename_blurr_dropout_earlystopping_batchnorm_regularization)

def build_dataset(majority_amount, augmentation=True, blur=False):
#majority_amount : maximum number of sunglasses examples in dataset
#augmentation boolean that does data augmentation with miroring effect on minority images if True
#blur boolean tat doex data augmentation with blurring effect on minority class
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

    def open_blurr_resize(path, liste):
        #resize an image loaded from path and adds it to a list such that it is ready for tensorflow (shape=(x,y,channel))
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)/255
        img = cv2.resize(img, (shape, shape))
        kernel_size= (3,3)
        img = cv2.blur(img, kernel_size)
        liste.append(img.reshape(shape, shape, channel))

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
            if blur:
                open_blurr_resize(path, image_set)
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

images, classes = build_dataset(3000, AUGMENTATION, BLURR)
print("<== Dataset is loaded  ==>")

#transform classes list to a binary matrix representation of the input so tensorflow can work with it
classes = tf.keras.utils.to_categorical(classes, num_classes=2)

images, classes = np.array(images), np.array(classes)

images_train, images_test, classes_train, classes_test = train_test_split(images, classes,
                                                                          test_size = TEST_PROPORTION)

def architecture(nb_dense, param_dense, nb_conv, param_conv, d_rate=0.3, f_rate=0.5):
#builds a model for a specified number of dense, conv layers
#with specified number of filter and nodes
#and dropout rate
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
            if REGULARIZATION == False:
                layers.append(tf.keras.layers.Dense(units=param_dense[j], activation=tf.nn.relu))
            else:
                layers.append(tf.keras.layers.Dense(units=param_dense[j],
                                                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                                    activation=tf.nn.relu))
            if DROPOUT:
                layers += [tf.keras.layers.Dropout(rate=f_rate)]

        layers.append(tf.keras.layers.Dense(units=2, activation=tf.nn.softmax))

        return tf.keras.Sequential(layers)

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='confusion matrix',
                        cmap='Blues',
                        filename="/confusion_matrix_basic.jpg"):
#plots and saves a confusion matrix with a given filename
#it is a heatmap inBlues to make things clearer
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(globalize("/best_models").format(layers_number, shape, shape) + filename)
    plt.show()

def get_flops(model_h5_path):
#computes floating points operations for a h5 model
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

def model_analysis(filename, output_filename, confusion_filename):
#takes excel file of F1 score of each trained model for an architecture (and specified anti-overfitting techniques)
#finds best model for a specified architecture
#saves & show a confusion matrix for this model with its flop as a title
    path_excel = globalize("/best_models").format(layers_number, shape, shape) + filename
    tab = pd.read_excel(open(path_excel, 'rb'), index_col=0).iloc[2:]
    tab = tab.rename(columns={'F1-score':'f1', 'F1-Score STD':'std',
                              'number of Conv2D':'conv',
                              'number of Dense':'dense',
                              'dropout rate for conv':'drc',
                              'dropout rate for dense': 'drd'})
    tab = np.array(tab.nlargest(1, 'f1').iloc[0])
    print(tab)
    conv_nb, dense_nb, f, d, batch_size, d_rate, f_rate = tuple(tab[0:7])

    model = architecture(dense_nb, [16], conv_nb, [8, 8//2], d_rate=0, f_rate=0)
    model.compile(optimizer = 'adam', loss = "binary_crossentropy")
    print("Model compiled")
    model.summary()
    model_file = globalize("/best_models/{}.h5".format(output_filename))

    # Callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                     mode='min', patience=2, min_lr=1e-12)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_file)

    model.fit(images_train, classes_train, batch_size = 4, epochs = epochs,
                                  validation_split=0.1, verbose = 2, shuffle = True,
                                  callbacks = [reduce_lr, checkpoint])
    flop = get_flops(model_file)
    print("The model has total flop of {}".format(flop))

    model = tf.keras.models.load_model(model_file)

    cm = confusion_matrix(y_true=np.argmax(classes_test, axis=1), y_pred=np.argmax(model.predict(images_test), axis=1))

    cm_plot_labels = ['No Sunglasses', 'Sunglasses']

    plot_confusion_matrix(cm=cm, classes=cm_plot_labels,
                          title="Total FLOP is: "+ str(flop), filename=confusion_filename)

#model_analysis(filename_basic, "model_min", "/confusion_matrix_min.jpg")
#model_analysis(filename_blurr_dropout_earlystopping_batchnorm_regularization,
#               "model_dropout_blurr_batchnorm_regularization",
#               "/confusion_matrix_dropout_blurr_batchnorm_regularization.jpg")