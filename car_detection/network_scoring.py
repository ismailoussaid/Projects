import numpy as np
import os
import cv2
import glob
import pandas as pd
from shapely.geometry import Polygon
from bbox_comparison import *

filename_test = "img_30.jpg"
global_path = "C:/Users/Ismail/Documents/Projects/Detect Cars/"
items = (0,-1)
thres = 0.1

def globalize(path, root = global_path):
    return root + path

images_path = sorted(glob.glob(globalize('dataset_car_detection/*.jpg')), key=os.path.getmtime)
tab_yolo = pd.read_csv(globalize("tab_yolo.csv"))
tab_resnet = pd.read_csv(globalize("tab_resnet.csv"))
tab_yolo_tiny = pd.read_csv(globalize("tab_yolo-tiny.csv"))

def iou(bbox_1, bbox_2):

    def transform(bbox):
        x1 = bbox['x1']
        y1 = bbox['y1']
        h = bbox['h']
        w = bbox['w']
        box = [[x1,y1], [x1+w,y1], [x1+w, y1+h], [x1, y1+h]]
        return box

    poly_1 = Polygon(transform(bbox_1))
    poly_2 = Polygon(transform(bbox_2))
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

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

def compare(sub_tab_1, sub_tab_2, methode='binary'):
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

        if methode == 'binary':
            if 0 not in same_detection:
                scores.append(1)
            else:
                scores.append(0)
        else:
            scores.append(max(same_detection))

    score = sum(scores)/len(scores)
    return True, score

def score(tab1, tab2, thres=thres, items=None):

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

if __name__ == '__main__':

    scores_1 = score(tab_yolo, tab_yolo_tiny, items=items)
    print(sum(scores_1)/len(scores_1))
    scores_2 = score(tab_yolo, tab_resnet, items=items)
    print(sum(scores_2)/len(scores_2))