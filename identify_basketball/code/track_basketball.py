import cv2
from score_basketballs import find_basketball
import numpy as np
import matplotlib.pyplot as plt


# load video
cap = cv2.VideoCapture('../videos/nash_shot_clean.mp4') # also: nash_cut.mp4

# read the first frame
ret, frame = cap.read()
frame_shape = frame.shape  # store the shape of the frame

point_of_interest = find_basketball(frame, initial_frame=True)

coordinates = []

# find basketball for the whole video
while cap.isOpened():
    
    ret, frame = cap.read()

    if not ret:
      break

    point_of_interest = find_basketball(frame, point_of_interest=point_of_interest, distance_weight=44)
    coordinates.append(point_of_interest)

cap.release()

# save the coordinates
coordinates = np.array(coordinates)
np.save('../data/basketball_trajectory_2.npy', coordinates)

# save in a csv file
np.savetxt('../data/basketball_trajectory_2.csv', coordinates, delimiter=',')


### PLOT THE TRAJECTORY ###
plt.plot(coordinates[:, 0], frame_shape[0] - coordinates[:, 1])
plt.title('Basketball Trajectory')
plt.savefig('../plots/basketball_trajectory_2.png')
plt.show()