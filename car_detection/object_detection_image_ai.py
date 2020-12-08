from imageai.Detection import ObjectDetection
import os
import glob
import pandas as pd
from utils import *
from argparse import ArgumentParser, SUPPRESS
#Requirements: tensorflow==1.13.0

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-m', '--model', help='Repository of h5 models', required=True, type=str)
    args.add_argument('-f', '--folder', help='Name of folder of images generated from video', required=True, type=str)
    args.add_argument('-n', '--tab_name', help='Name of csv tab', required=True, type=str)
    args.add_argument('-t', '--threshold', help='Threshold for custom object detector', required=True, type=int, default=75)
    args.add_argument('-i', '--items', help='Number of items is maximum if None', required=True, default=None)
    return parser

def detect_imageai(label, threshold, items, input_path, tab_name, detection_output_folder="results/"):
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

    output_folder = globalize(folder+'/')+detection_output_folder
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
            filename = path[len(globalize(folder) + "//")-1:]
            multiple_append([x1, y1, ws, hs, objects, filenames, probas],
                            [x_min, y_min, w, h, object, filename, proba])

            tab = pd.DataFrame(data=d)
            tab.to_csv(globalize(tab_name + f"_{label}.csv"))

    return tab

if __name__ == '__main__':
    networks = ["yolo-tiny", "resnet", "yolo"]
    args = build_argparser().parse_args()

    model_yolo = globalize(args.model + "/yolo_v3.h5")
    model_resnet = globalize(args.model + "/resnet50_coco_best_v2.1.0.h5")
    model_yolo_tiny = globalize(args.model + "vehicle_detection/yolo-tiny.h5")
    images_folder = globalize(f'{args.folder}/*.jpg')

    for network in networks:
        print(f"{network} is detecting objects")
        detect_imageai(label=network, threshold=args.threshold, items=args.items, input_path=images_folder, tab_name=args.tab_name)