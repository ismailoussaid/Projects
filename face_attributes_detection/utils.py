
import os
import numpy as np

def multiple_append(listes, elements):
    # append different elements to different lists in the same time
    if len(listes) != len(elements):
        # ensure, there is no mistake in the arguments
        raise RuntimeError("lists and elements do not have the same length")
    else:
        for l, e in zip(listes, elements):
            l.append(e)


def adapt(x):
    return (x+1)/2


def anti_adapt(x):
    return (-x+1)/2

def avg(liste):
    if type(liste)!=list:
        print("it is not a list")
    else:
        return sum(liste)/len(liste)


def build_folder(path):
    # build a directory and confirm execution with a message
    try:
        os.makedirs(path)
        print("<== {} directory created ==>")
    except OSError:
        print("Creation of the directory %s failed" % path)


def labelize(outputs):
    # transform the vector output of a final dense layer with softmax
    # the most likely label gets one and the other takes 0
    # as would .utils.to_categorical do to a binary categorical attributes
    return np.argmax(outputs, axis=1)


def pred_to_label(prediction, attribute):
    if attribute == 'mustache':
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
