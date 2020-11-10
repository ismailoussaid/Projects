from imageai.Detection import ObjectDetection
import numpy as np
import os
import cv2
import glob
import pandas as pd
from shapely.geometry import Polygon

items = (0,-1)
filename_test = "img_30.jpg"
threshold = 75
thres = 0.1
global_path = "C:/Users/Ismail/Documents/Projects/Detect Cars/"
shape, channel = 608, 3

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

def detect_imageai(label = "yolo", thres=threshold, items=(28,32)):
    detector = ObjectDetection()

    if label == "yolo":
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(model_yolo)
    elif label == "resnet":
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(model_resnet)
    else:
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
    return tab

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