
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