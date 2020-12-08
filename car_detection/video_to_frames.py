import cv2
from utils import *
from argparse import ArgumentParser

global_path = "C:/Users/Ismail/Documents/Projects/Detect Cars/"

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-v', '--video', help='Path to video file', required=True, type=str)
    args.add_argument('-o', '--output_folder', help='output folder of frames', required=True, type=str)
    args.add_argument('-f', '--freq', default=1, type=int)
    return parser

if __name__ == '__main__':
    args = build_argparser().parse_args()
    print(args.video)
    # Opens the Video file
    cap = cv2.VideoCapture(globalize(args.video))
    folder = globalize(args.output_folder)
    build_folder(folder)
    i = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = frame
        if i%args.freq == 0:
            cv2.imwrite(folder + 'img_' + str(i//args.freq) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()