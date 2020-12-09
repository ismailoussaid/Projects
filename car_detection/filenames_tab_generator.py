import os
import pandas as pd
from utils import *
import glob

root_path = 'C:/frames/'
#folder where to find images
folder = '30343_gaussian_25/results'
#network label to name the final tab
label = '30343_gaussian_25_person-detection-retail-0013'
#path of frames
images_folder = globalize(f'{folder}/*.jpg', root=root_path)
#list of all frames filenames
images_path = sorted(glob.glob(images_folder), key=os.path.getmtime)
images_path = [img for img in images_path if "person-detection-retail-0013" in img]
filenames = []

for path in images_path:
    filename = path[len(globalize(folder, root=root_path) + "//") - 1:]
    filenames.append(filename)
    d = {'file':filenames}
    tab = pd.DataFrame(data=d)
    tab.to_csv(globalize(f"filenames_{label}.csv", root=root_path))
