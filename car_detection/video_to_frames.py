import cv2
from utils import *

global_path = "C:/Users/Ismail/Documents/Projects/Detect Cars/"

# Opens the Video file
cap = cv2.VideoCapture(globalize('rainy-evening-on-airline-highway-traffic-webcam.mp4'))
folder = globalize('rainy_frames/')
build_folder(folder)
i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frame = frame
    cv2.imwrite(folder + 'img_' + str(i) + '.jpg', frame)
    i += 1

cap.release()
cv2.destroyAllWindows()