import cv2
from utils import *

global_path = "C:/Users/Ismail/Documents/Projects/Detect Cars/"
freq = 5

# Opens the Video file
cap = cv2.VideoCapture(globalize('shanghai-traffic-rainy-afternoon.mp4'))
folder = globalize('rainy_frames_2/')
build_folder(folder)
i = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frame = frame
    if i%5 == 0:
        cv2.imwrite(folder + 'img_' + str(i//freq) + '.jpg', frame)
    i += 1

cap.release()
cv2.destroyAllWindows()