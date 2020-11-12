from imageai.Detection import ObjectDetection
import numpy as np
import os
import cv2
import glob
import pandas as pd
import time

items = (0,-1)

filename_test = "img_30.jpg"
threshold = 75
thres = 0.1
global_path = "C:/Users/Ismail/Documents/Projects/Detect Cars/"
shape, channel = 608, 3
networks = ["yolo-tiny", "resnet", "yolo"]

def globalize(path, root = global_path):
    return root + path

model_yolo = globalize("yolo_v3.h5")
model_resnet = globalize("resnet50_coco_best_v2.1.0.h5")
model_yolo_tiny = globalize("yolo-tiny.h5")
image_input = globalize("image_test.JPG")
image_output = globalize("Detect Cars/output_test.jpg")
images_path = sorted(glob.glob(globalize('dataset_car_detection/*.jpg')), key=os.path.getmtime)

def build_folder(path):
    # build a directory and confirm execution with a message
    try:
        os.makedirs(path)
        print("<== {} directory created ==>")
    except OSError:
        print("Creation of the directory %s failed" % path)

def multiple_append(listes, elements):
    # append different elements to different lists in the same time
    if len(listes) != len(elements):
        # ensure, there is no mistake in the arguments
        raise RuntimeError("lists and elements do not have the same length")
    else:
        for l, e in zip(listes, elements):
            l += [e]

def random_crop(frame, shape=608, channel=3):
    original_height = frame.shape[0]
    max_x = frame.shape[1] - original_height
    random_x = np.random.randint(0, max_x)
    im = frame[:, random_x:random_x + original_height]
    im = cv2.resize(im, (shape, shape)).reshape((shape, shape, channel))
    return im

def detect_imageai(label = "yolo", thres=threshold, items=items):
    detector = ObjectDetection()

    if label == "yolo":
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(model_yolo)
    elif label == "resnet":
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(model_resnet)
    elif label == "yolo-tiny":
        detector.setModelTypeAsTinyYOLOv3()
        detector.setModelPath(model_yolo_tiny)

    detector.loadModel()
    custom = detector.CustomObjects(car=True, truck=True, motorcycle=True, bus=True)

    output_folder = globalize("outputs/")
    build_folder(output_folder)

    x1, y1, ws, hs, filenames, objects, probas = [], [], [], [], [], [], []
    d = {'file':filenames, 'object': objects, 'x1':x1, 'y1':y1, 'w':ws, 'h':hs, 'p':probas}

    if items == None:
        a,b = 0,-1
    else:
        a,b = items

    i = a

    for path in images_path[a:b]:

        output_path = output_folder + f'detection_{label}_{i}.jpg'
        i+=1

        detection = detector.detectCustomObjectsFromImage(input_image=path,
                                                                    custom_objects=custom,
                                                                    output_image_path=output_path,
                                                                    minimum_percentage_probability=thres)

        for item in detection:
            box = item["box_points"]
            x_min = box[0]
            x_max = box[2]
            y_min = box[1]
            y_max = box[3]
            w = x_max - x_min
            h = y_max - y_min
            object = item["name"]
            proba = item['percentage_probability']
            filename = path[len("C:/Users/Ismail/Documents/Projects/Detect Cars/dataset_car_detection//")-1:]
            multiple_append([x1, y1, ws, hs, objects, filenames, probas],
                            [x_min, y_min, w, h, object, filename, proba])

    tab = pd.DataFrame(data=d)
    tab.to_csv(globalize(f"tab_{label}.csv"))
    return tab

if __name__ == '__main__':
    for network in networks:
        print(f"{network} is detecting objects")
        start = time.time()
        detect_imageai(label=network)
        end = time.time()
        print(f"detections completed for {network} in {end-start} seconds")