import cv2
from utils import *
from argparse import ArgumentParser

global_path = "C:/frames/"

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-v', '--video', help='Path to video file', required=True, type=str)
    args.add_argument('-o', '--output_folder', help='Name of the output folder of frames', required=True, type=str)
    args.add_argument('-f', '--freq', default=1, type=int)
    args.add_argument('-b', '--blur', default=1, help='Size of average filter', required=True, type=int)
    args.add_argument('-t', '--type', default='average', help='Type of filter', required=True, type=str)
    return parser

if __name__ == '__main__':
    args = build_argparser().parse_args()
    type, blur = args.type, args.blur

    # Opens the Video file
    cap = cv2.VideoCapture(globalize(args.video))

    if args.blur != 1:
        folder = globalize(args.output_folder + f'_{type}_{blur}', root = global_path)
    else:
        folder = globalize(args.output_folder, root = global_path)

    build_folder(folder)
    i = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = frame
        if i%args.freq == 0:
            size = (blur, blur)
            if type == 'average':
                frame = cv2.blur(frame, size)
            elif type == 'gaussian':
                frame = cv2.GaussianBlur(frame, size, 0)
            cv2.imwrite(folder + '/img_' + str(i//args.freq) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()