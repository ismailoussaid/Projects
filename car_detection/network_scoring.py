from imageai.Detection import ObjectDetection
import numpy as np
import os
import cv2
import glob
import pandas as pd
from shapely.geometry import Polygon
from bbox_comparison import *

items = (0,-1)
filename_test = "img_30.jpg"
threshold = 75
thres = 0.1
global_path = "/Detect Cars/"
shape, channel = 608, 3

def globalize(path, root = global_path):
    return root + path

model_yolo = globalize("../../Detect Cars/yolo_v3.h5")
model_resnet = globalize("../../Detect Cars/resnet50_coco_best_v2.1.0.h5")
model_yolo_tiny = globalize("../../Detect Cars/yolo-tiny.h5")
image_input = globalize("../../Detect Cars/image_test.JPG")
image_output = globalize("Detect Cars/output_test.jpg")
images_path = sorted(glob.glob(globalize('dataset_car_detection/*.jpg')), key=os.path.getmtime)

def cluster_subtab(sub_tab, threshold=1e-1):
    df = sub_tab.to_dict('records')

    if len(df) == 0:
        return []

    else:
        objects = []

        for k in range(len(df)-1):

            data = df[k]
            list_iou = []

            for i in range(k+1,len(df)):
                obj = df[i]
                list_iou.append(iou(data, obj))

            if list_iou == [0]*len(list_iou):
                objects.append(data)

        objects.append(df[-1])

        clusters = []
        for i in range(len(objects)):
            cluster = []
            object = objects[i]
            for k in range(len(df)):
                element = df[k]
                if iou(object, element) > threshold:
                    cluster.append(element)
            clusters.append(cluster)

        return clusters

def avg_cluster(clusters):
    detections = []

    if len(clusters)==0:
        return clusters

    else:
        for cluster in clusters:
            avgDict = {}

            for key, v in cluster[0].items():
                if type(v) == str:
                    avgDict[key] = v
                else:
                    avgDict[key] = 0

            for i in range(len(cluster)):
                for key, v in avgDict.items():
                    if type(v) != str:
                        avgDict[key] += int(cluster[i][key])

            for key, v in avgDict.items():
                if type(v) != str:
                    avgDict[key] = round(v/len(cluster))

            detections.append(avgDict)
    return detections

def compare(sub_tab_1, sub_tab_2):
    scores = []

    if sub_tab_1 == []:
        #print("no detection in strong learner")
        return 0, 0

    if sub_tab_2 == []:
        #print("no detection in weak learner")
        return 0, 0

    for k in range(len(sub_tab_1)):
        detection_ref = sub_tab_1[k]
        same_detection = []

        for j in range(len(sub_tab_2)):
            same_detection.append(iou(detection_ref, sub_tab_2[j]))

        scores.append(max(same_detection))

    score = sum(scores)/len(scores)
    return True, score

def score(network1 = "yolo", network2 = "resnet", thres=thres, items=None):
    tab1 = detect_imageai(network1, items = items)
    tab2 = detect_imageai(network2, items = items)

    if items == None:
        a,b = 0,-1
    else:
        a,b = items

    filenames = images_path[a:b]
    scores = []

    for path in filenames:
        filename = path[len("C:/Users/Ismail/Documents/Projects/Detect Cars/dataset_car_detection//")-1:]

        subtab_1 = avg_cluster(cluster_subtab(tab1[tab1.file == filename], threshold=thres))
        subtab_2 = avg_cluster(cluster_subtab(tab2[tab2.file == filename], threshold=thres))

        boolean, score = compare(subtab_1, subtab_2)
        scores.append(score)

    return scores

scores_1 = score("yolo", "yolo-tiny", items=items)
print(scores_1)
print(sum(scores_1)/len(scores_1))
scores_2 = score("yolo", "resnet", items=items)
print(scores_2)
print(sum(scores_2)/len(scores_2))