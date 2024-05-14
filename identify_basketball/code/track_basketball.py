import cv2
from score_basketballs import find_basketball
import numpy as np
import matplotlib.pyplot as plt



# load video
cap = cv2.VideoCapture('../videos/nash_miss_fullcut.mp4')

# read the first frame
ret, frame = cap.read()

point_of_interest = find_basketball(frame, initial_frame=True)

# for each frame in the video
while cap.isOpened():
   
   
   ret, frame = cap.read()
   point_of_interest = find_basketball(frame, initial_frame=False, point_of_interest=point_of_interest, distance_weight=25)


   if cv2.waitKey(1) & 0xFF == ord('q'):
      break
   
cap.release()

   