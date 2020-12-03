
import pandas as pd
from utils import *

global_path = "C:/Users/Ismail/Documents/Projects/Detect Cars/"
items = None

def globalize(path, root = global_path):
    return root + path

tab_yolo = pd.read_csv(globalize("tab_yolo.csv"))
tab_resnet = pd.read_csv(globalize("tab_resnet.csv"))
tab_yolo_tiny = pd.read_csv(globalize("tab_yolo-tiny.csv"))
tab_vehicle_detection_adas_0002 = pd.read_csv(globalize("tab_vehicle-detection-adas-0002.csv"))
tab_vehicle_detection_adas_binary_0001 = pd.read_csv(globalize("tab_vehicle-detection-adas-binary-0001.csv"))
tab_person_vehicle_bike_detection_crossroad_0078 = pd.read_csv(globalize("tab_person-vehicle-bike-detection-crossroad-0078.csv"))
tab_person_vehicle_bike_detection_crossroad_1016 = pd.read_csv(globalize("tab_person-vehicle-bike-detection-crossroad-1016.csv"))
tab_pedestrian_and_vehicle_detector_adas_0001 = pd.read_csv(globalize("tab_pedestrian-and-vehicle-detector-adas-0001.csv"))

tabs = [tab_yolo, tab_resnet, tab_yolo_tiny,
        tab_vehicle_detection_adas_0002, tab_vehicle_detection_adas_binary_0001,
        tab_person_vehicle_bike_detection_crossroad_0078, tab_person_vehicle_bike_detection_crossroad_1016,
        tab_pedestrian_and_vehicle_detector_adas_0001]
names = ["yolo", "resnet", "yolo_tiny",
        "vehicle_detection_adas_0002", "vehicle_detection_adas_binary_0001",
        "person_vehicle_bike_detection_crossroad_0078", "person_vehicle_bike_detection_crossroad_1016",
        "pedestrian_and_vehicle_detector_adas_0001"]

def iou(boxA, boxB):

    xA = max(boxA['x1'], boxB['x1'])
    yA = max(boxA['y1'], boxB['y1'])
    xB = min(boxA['x1']+boxA['w'], boxB['x1']+boxA['w'])
    yB = min(boxA['y1']+boxA['h'], boxB['y1']+boxB['h'])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA+1, 0)) * max((yB - yA+1, 0)))

    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA['w']+1) * (boxA['h']+1))
    boxBArea = abs((boxB['h']+1) * (boxB['h']+1))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
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
    #tp is the amount of vehicles detected by network1 and by network 2
    #tn is the amount of elements not identified as vehicles by network1 but detected by network 2
    #fn is the amount of vehicles detected by network1 and not by network 2  100
    
    n1, n2 = len(sub_tab_1), len(sub_tab_2)
    tp, tn, fn = 0, 0, 0

    if n1==0 and n2>0:
        tn = n2
        return tp, tn, fn

    elif n2==0 and n1>0:
        fn = n1
        return tp, tn, fn

    elif n1==0 and n2==0:
        tp = 1
        return tp, tn, fn

    else:
        deleted_s1 = set()
        deleted_s2 = set()

        for k in range(len(sub_tab_1)):
            for j in range(len(sub_tab_2)):
                if k in deleted_s1 or j in deleted_s1:
                    continue
                area = iou(sub_tab_1[k], sub_tab_2[j])
                if area >= comparison_threshold:
                    tp += 1
                    deleted_s1.add(k)
                    deleted_s2.add(j)

        sub_tab_1 = [x for idx, x in enumerate(sub_tab_1) if idx not in deleted_s1]
        sub_tab_2 = [x for idx, x in enumerate(sub_tab_2) if idx not in deleted_s2]

        fn = len(sub_tab_1)
        tn = len(sub_tab_2)

        """
        total_amount = tp+tn+fn

        tp *= 100/total_amount
        tn *= 100/total_amount
        fn *= 100/total_amount
        """

        return tp, tn, fn

def score(tab1, tab2, averaging_threshold, comparison_threshold=0.2, scr='scoring', methode = binary, items=items):
    if items == None or items == (0, -1):
        a,b = 0,-1
    else:
        a,b = items

    filenames = tab1['file']
    #all distinct filename
    filenames = filenames.drop_duplicates().iloc[a:b]

    scores = []
    tp_scores, tn_scores, fp_scores = [], [], []

    for filename in filenames:
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
        return sum(scores)
    else:
        return sum(tp_scores), sum(tn_scores), sum(fp_scores)

if __name__ == '__main__':
    s = 0.1
    names1, names2, tps, fps, fns = [], [], [], [], []
    data = {'network 1': names1, 'network 2': names2, 'tp': tps, 'fp': fps, 'fn': fns}
    n = len(names)
    for i in range(n):
        for j in range(n):
            name1, name2, tab1, tab2 = names[i], names[j], tabs[i], tabs[j]
            if name1 != name2:
                tp, fp, fn = score(tab1, tab2, s, scr='confusion')
                multiple_append([names1, names2, tps, fps, fns], [name1, name2, tp, fp, fn])
                tab = pd.DataFrame(data=data)
                tab.to_csv(globalize('scores_comparative.csv'))