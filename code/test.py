import cv2
from score_basketballs import track_basketball


# load video
cap = cv2.VideoCapture('../videos/nash_shot_clean.mp4') # also: nash_cut.mp4

# display and saves the basketball trajectory
coordinates = track_basketball(cap, plot_save_file='../plots/basketball_trajectory_3', csv_save_file='../data/basketball_trajectory_3')