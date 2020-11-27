import numpy as np
import os
import cv2
import glob
import pandas as pd
from shapely.geometry import Polygon
from bbox_comparison import *

filename_test = "img_250.jpg"
global_path = "C:/Users/Ismail/Documents/Projects/Detect Cars/"
sample_index = 333
#items=None allows to test networks for every image
items = (sample_index,1+sample_index)

def globalize(path, root = global_path):
    return root + path

images_path = sorted(glob.glob(globalize('dataset_car_detection/*.jpg')), key=os.path.getmtime)
print(images_path[10])
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

def cluster_subtab(sub_tab, threshold):
    df = sub_tab.to_dict('records')
    n = len(df)

    if n == 0:
        return []

    if n == 1:
        return [df]

    objects = []

    for k in range(n-1):

        data = df[k]
        list_iou = []

        for i in range(k+1,n):
            obj = df[i]
            list_iou.append(iou(data, obj))

        boolean = [x < threshold for x in list_iou]
        if all(boolean): #== [0]*len(list_iou):
            objects.append(data)

    objects.append(df[-1])

    clusters = []
    for i in range(len(objects)):
        cluster = []
        object = objects[i]
        for k in range(len(df)):
            element = df[k]
            area = iou(object, element)
            if area > threshold:
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
            p = len(cluster)
            for key, v in cluster[0].items():
                if type(v) == str:
                    avgDict[key] = v
                else:
                    avgDict[key] = 0

            for i in range(p):
                for key, v in avgDict.items():
                    if type(v) != str:
                        avgDict[key] += int(cluster[i][key])

            for key, v in avgDict.items():
                if type(v) != str:
                    avgDict[key] = round(v/p)

            detections.append(avgDict)

    return detections

def binary(liste):
    if 0 not in liste:
        return 1
    else:
        return 0

def overlap(liste):
    return max(liste)

def compare(sub_tab_1, sub_tab_2, methode=binary):
    n1, n2 = len(sub_tab_1), len(sub_tab_2)
    scores = []

    if n1 == 0:
        #print("no detection in strong learner")
        return 0

    elif n2 == 0:
        #print("no detection in weak learner")
        return 0

    else:
        for k in range(n1):
            detection_ref = sub_tab_1[k]
            same_detection = []
            for j in range(n2):
                same_detection.append(iou(detection_ref, sub_tab_2[j]))
            scores.append(methode(same_detection))

        score = sum(scores)/len(scores)
        return score

def confusion_compare(sub_tab_1, sub_tab_2, comparison_threshold):
    #tp is the amount of vehicles detected by network1 and by network 2 0
    #tn is the amount of elements not identified as vehicles by network1 but detected by network 2  20
    #fn is the amount of vehicles detected by network1 and not by network 2  100
    
    n1, n2 = len(sub_tab_1), len(sub_tab_2)
    tp, tn, fn = 0, 0, 0

    if n1==0 and n2>0:
        #print("no detection in strong learner")
        tn = 100
        return tp, tn, fn

    elif n2==0 and n1>0:
        #print("no detection in weak learner")
        fn = 100
        return tp, tn, fn

    elif n1==0 and n2==0:
        #print("no detection in both learners")
        tp = 100
        return tp, tn, fn

    else:
        s1 = sub_tab_1.copy()
        s2 = sub_tab_2.copy()

        for k in s1:
            for j in s2:
                area = iou(k,j)
                if area >= comparison_threshold:
                    tp += 1
                    sub_tab_1.remove(k)
                    sub_tab_2.remove(j)

        fn = len(sub_tab_1)
        tn = len(sub_tab_2)

        #total_amount = tp+tn+fn
        total_amount = n1

        tp *= 100/total_amount
        tn *= 100/total_amount
        fn *= 100/total_amount
        return tp, tn, fn

def score(tab1, tab2, averaging_threshold, comparison_threshold=0.2, scr='scoring', methode = binary, items=items):
    if items == None or items == (0, -1):
        a,b = 0,-1
    else:
        a,b = items

    filenames = images_path[a:b]
    scores = []
    tp_scores, tn_scores, fp_scores = [], [], []

    for path in filenames:
        filename = path[len("C:/Users/Ismail/Documents/Projects/Detect Cars/dataset_car_detection//")-1:]
        print(path)

        subtab_1 = avg_cluster(cluster_subtab(tab1[tab1.file == filename],
                                              threshold=averaging_threshold))
        subtab_2 = avg_cluster(cluster_subtab(tab2[tab2.file == filename],
                                              threshold=averaging_threshold))
        if scr == 'scoring':
            score = compare(subtab_1, subtab_2, methode)
            scores.append(score)
        else:
            tp, tn, fp = confusion_compare(subtab_1, subtab_2, comparison_threshold=comparison_threshold)
            tp_scores.append(tp)
            tn_scores.append(tn)
            fp_scores.append(fp)

    if scr == 'scoring':
        return sum(scores)/len(scores)
    else:
        n = len(tp_scores)
        return sum(tp_scores)/n, sum(tn_scores)/n, sum(fp_scores)/n

if __name__ == '__main__':
    s = 0.1
    meth = binary
    t1, t2 = tab_yolo_tiny, tab_resnet
    #print(score(tab_yolo, t1, s, scr='confusion'))
    print(score(tab_yolo, t2, s, scr='confusion'))