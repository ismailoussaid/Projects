import cv2
from utils import *
from argparse import ArgumentParser
import pandas as pd

global_path = "C:/frames/"

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-f', '--folder', help='Name of the folder of frames', required=True, type=str)
    args.add_argument('-o', '--output', help='Name of the output path of the video', required=True, type=str)
    args.add_argument('-t', '--tab', help='Name of the tab containing filename in good order', required=True, type=str)
    return parser

if __name__ == '__main__':
    args = build_argparser().parse_args()

    img_array = []
    i = 0
    for filename in pd.read_csv(globalize(args.tab, root=global_path))['file']:
        img = cv2.imread(globalize(args.folder, root=global_path)+filename)
        height, width, layers = img.shape
        size = (width, height)
        if i == 0:
            out = cv2.VideoWriter(globalize(args.output + '.avi', root=global_path), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        if i % 50 == 0:
            print(f"image n°{i} is put in the video")
        i += 1
        out.write(img)

    out.release()