from imageai.Detection import ObjectDetection
import os
import glob
import pandas as pd
import time
from utils import *

#Requirements: tensorflow==1.13.0
items = (0,-1)
threshold = 75
shape, channel = 608, 3
networks = ["yolo", "yolo-tiny", "resnet"]

model_yolo = globalize("yolo_v3.h5")
model_resnet = globalize("resnet50_coco_best_v2.1.0.h5")
model_yolo_tiny = globalize("yolo-tiny.h5")
image_output = globalize("Detect Cars/output_test.jpg")
images_folder = globalize('rainy_frames/*.jpg')

def detect_imageai(label = "yolo", threshold=threshold, items=items, input_path=images_folder ,detection_output_folder="results/"):
    detector = ObjectDetection()
    images_path = sorted(glob.glob(input_path), key=os.path.getmtime)

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

    output_folder = globalize('rainy_frames/')+detection_output_folder
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
                                                                    minimum_percentage_probability=threshold)

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
            filename = path[len("C:/Users/Ismail/Documents/Projects/Detect Cars/rainy_frames//")-1:]
            multiple_append([x1, y1, ws, hs, objects, filenames, probas],
                            [x_min, y_min, w, h, object, filename, proba])

            tab = pd.DataFrame(data=d)
            tab.to_csv(globalize(f"tab_rainy_{label}.csv"))

    return tab

if __name__ == '__main__':
    for network in networks:
        print(f"{network} is detecting objects")
        start = time.time()
        detect_imageai(label=network)
        end = time.time()
        print(f"detections completed for {network} in {end-start} seconds")