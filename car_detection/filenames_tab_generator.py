import os
import pandas as pd
from utils import *
import glob

folder = 'dataset_car_detection'
images_folder = globalize(f'{folder}/*.jpg')
images_path = sorted(glob.glob(images_folder), key=os.path.getmtime)
filenames = []

for path in images_path:
    filename = path[len(globalize(folder) + "//") - 1:]
    filenames.append(filename)
    d={'file':filenames}
    tab = pd.DataFrame(data=d)
    tab.to_csv(globalize(f"filenames_{folder}.csv"))
    print(filename)