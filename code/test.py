import cv2
from score_basketballs import track_basketball
import numpy as np
from scipy.spatial import distance
from presentation import superimpose_trajectory

# load video
cap = cv2.VideoCapture('../videos/nash_shot.mp4')

# Example ARR1 and ARR2
ARR1 = np.array([(1, 2), (3, 4), (5, 6)])
ARR2 = np.array([(2, 2), (4, 6), (10, 12), (7, 8), (54, 33), (54, 34)])

# Define the distance threshold
threshold = 5

# Compute the distance from each point in ARR2 to each point in ARR1
dists = distance.cdist(ARR2, ARR1, 'euclidean')

# Find the minimum distance to any point in ARR1 for each point in ARR2
min_dists = np.min(dists, axis=1)

# Find the indices where the distance is greater than the threshold
indices = np.where(min_dists > threshold)[0]

# Get the points in ARR2 that are farther than the threshold
far_points = ARR2[indices]

print("Indices of points farther than the threshold:", indices)
print("Points in ARR2 farther than the threshold:", far_points)

# display and saves the basketball trajectory
coordinates = track_basketball(
   cap, video='nash_shot', save_csv=False, save_plot=False, show_plot=True)

# superimpose the basketball trajectory on the image
image = cv2.imread('../images/nash_shooting.png')
image = np.array(image)

# superimpose the trajectory on the image
image_with_trajectory = superimpose_trajectory(image, coordinates)