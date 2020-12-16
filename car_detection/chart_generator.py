import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from argparse import ArgumentParser
import re

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-i', '--index', help='Index of the name of the reference model', required=True, type=int)
    args.add_argument('-l', '--label', help='Label of the dataset or analysis', required=True, type=str)
    args.add_argument('-r', '--root_path', help='Path of the folder where to put and load files', required=True, type=str)
    return parser

if __name__ == '__main__':
    args = build_argparser().parse_args()
    barWidth = 0.4
    fs = 16
    root_path = args.root_path
    label = args.label

    tab = pd.read_csv(globalize(f"scores_comparative_{label}.csv", root=root_path))
    names = ["yolo", "resnet", "yolo_tiny",
             "vehicle_detection_adas_0002", "vehicle_detection_adas_binary_0001",
             "pedestrian_and_vehicle_detector_adas_0001", "person_vehicle_bike_detection_crossroad_0078",
             "person_vehicle_bike_detection_crossroad_1016", "person_vehicle_bike_detection_crossroad_1016-FP16"]
    chart_names = ["YOLO v3", "Resnet 50", "Tiny YOLO",
                   "Vehicle Adas 0002", "Adas Binary 0001",
                   "Adas 0001", "Crossroad 0078",
                   "Crossroad 1016", "Crossroad 1016 FP-16"]
    chart_names_new = [chart_name.replace(' ', '\n') for chart_name in chart_names]

    N1 = chart_names[args.index]
    model = names[args.index]
    flops = [65.3, 4.3, 5.8,
             2.79, 0.94,
             3.97, 3.96,
             3.56, 3.56]

    indexes = []
    for name in names:
        sample = tab[tab['network 2']==name].reset_index().drop('level_0', axis=1)
        sample = sample[sample['network 1'] == model]
        indexes.append(sample.iloc[0]['index'])

    flops_norm = [flops[k]/max(flops) for k in range(len(flops))]
    ratio = [indexes[k]**2/flops_norm[k] for k in range(len(flops_norm))]

    #Bar plot of the performance measure
    plt.figure(figsize=(17,12))
    plt.xticks(fontsize=fs)
    bar = plt.bar(chart_names_new, indexes, width = barWidth, color = ['deepskyblue' for _ in names],
               edgecolor = ['black' for _ in names], linewidth = 2)
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, round(height, 3), ha='center', va='bottom', fontsize=fs)

    plt.title(f'Measure I({N1}, Model)', fontsize=int(fs*1.25))
    plt.savefig(globalize(f"chart_{label}_yolo_reference.jpg", root=root_path))

    #Bar plot of the ratio measure
    plt.figure(figsize=(17, 12))
    plt.xticks(fontsize=fs)
    bar = plt.bar(chart_names_new, ratio, width=barWidth, color=['deepskyblue' for _ in names],
                  edgecolor=['black' for _ in names], linewidth=2)

    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, round(height, 2), ha='center', va='bottom', fontsize=fs)

    plt.title(f'Ratio I({N1}, Model)/FLOPS', fontsize=int(fs*1.25))
    plt.savefig(globalize(f"chart_{label}_{model}_reference_norm.jpg", root=root_path))