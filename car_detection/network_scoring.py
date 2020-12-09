
import pandas as pd
from utils import *
from shapely.geometry import Polygon
from argparse import ArgumentParser

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-f', '--filenames', help='Tab of filenames', required=True, type=str)
    args.add_argument('-t', '--tab_base', help='Name of the csv tab', required=True, type=str)
    args.add_argument('-i', '--items', help='Number of items is maximum if None', required=True, default=None)
    return parser

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
            area = iou(data, obj)
            list_iou.append(area)

        boolean = [x < threshold for x in list_iou]
        if all(boolean):
            objects.append(data)

    objects.append(df[-1])

    clusters = []
    for i in range(len(objects)):
        object = objects[i]
        cluster = [object]
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
        return 0

    elif n2 == 0:
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
        return tp, tn, fn

    else:
        deleted_s1 = set()
        deleted_s2 = set()

        for k in range(len(sub_tab_1)):
            for j in range(len(sub_tab_2)):
                if k in deleted_s1 or j in deleted_s2:
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

        return tp, tn, fn

def score(tab1, tab2, averaging_threshold, filenames=filenames, comparison_threshold=0.2, scr='scoring', methode = binary, items=items):
    if items == None or items == (0, -1):
        a,b = 0,-1
    else:
        a,b = items

    #all distinct filename
    filenames = filenames.iloc[a:b]['file']
    scores = []
    tp_scores, tn_scores, fp_scores = [], [], []

    for filename in filenames:
        tab_1 = tab1[tab1.file == filename]
        tab_2 = tab2[tab2.file == filename]
        subtab_1 = avg_cluster(cluster_subtab(tab_1,
                                              threshold=averaging_threshold))
        subtab_2 = avg_cluster(cluster_subtab(tab_2,
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

    args = build_argparser().parse_args()

    filenames = pd.read_csv(globalize(f"filenames_{args.filenames}.csv"))
    tab_yolo = pd.read_csv(globalize(f"{args.tab_base}_yolo.csv"))
    tab_resnet = pd.read_csv(globalize(f"{args.tab_base}_resnet.csv"))
    tab_yolo_tiny = pd.read_csv(globalize(f"{args.tab_base}_yolo-tiny.csv"))
    tab_vehicle_detection_adas_0002 = pd.read_csv(globalize(f"{args.tab_base}_vehicle-detection-adas-0002.csv"))
    tab_vehicle_detection_adas_binary_0001 = pd.read_csv(
        globalize(f"{args.tab_base}_vehicle-detection-adas-binary-0001.csv"))
    tab_person_vehicle_bike_detection_crossroad_0078 = pd.read_csv(
        globalize(f"{args.tab_base}_person-vehicle-bike-detection-crossroad-0078.csv"))
    tab_person_vehicle_bike_detection_crossroad_1016 = pd.read_csv(
        globalize(f"{args.tab_base}_person-vehicle-bike-detection-crossroad-1016.csv"))
    tab_pedestrian_and_vehicle_detector_adas_0001 = pd.read_csv(
        globalize(f"{args.tab_base}_pedestrian-and-vehicle-detector-adas-0001.csv"))

    tabs = [tab_yolo, tab_resnet, tab_yolo_tiny,
            tab_vehicle_detection_adas_0002, tab_vehicle_detection_adas_binary_0001,
            tab_person_vehicle_bike_detection_crossroad_0078, tab_person_vehicle_bike_detection_crossroad_1016,
            tab_pedestrian_and_vehicle_detector_adas_0001]
    names = ["yolo", "resnet", "yolo_tiny",
             "vehicle_detection_adas_0002", "vehicle_detection_adas_binary_0001",
             "person_vehicle_bike_detection_crossroad_0078", "person_vehicle_bike_detection_crossroad_1016",
             "pedestrian_and_vehicle_detector_adas_0001"]

    s = 0.1
    names1, names2, tps, fps, fns = [], [], [], [], []
    data = {'network 1': names1, 'network 2': names2, 'tp': tps, 'fp': fps, 'fn': fns}
    n = len(names)
    for i in range(n-1):
        for j in range(i+1,n):
            name1, name2, tab1, tab2 = names[i], names[j], tabs[i], tabs[j]
            tp, fp, fn = score(tab1, tab2, s, scr='confusion')
            multiple_append([names1, names2, tps, fps, fns], [name1, name2, tp, fp, fn])
            tab = pd.DataFrame(data=data)
            tab.to_csv(globalize(f'scores_comparative_{args.tab_base[4:]}.csv'))