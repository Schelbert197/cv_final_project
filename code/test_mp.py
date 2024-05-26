from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial import procrustes
import numpy as np
from track_motion import track_shot, find_release
from score_basketballs import track_basketball
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Set backend to Agg


def compare_lines_procrustes(line1, line2):
    # Convert lines to numpy arrays
    line1 = np.array(line1)
    line2 = np.array(line2)

    # Pad the shorter line with repeated last points to make them equal in length
    if len(line1) < len(line2):
        line1 = np.vstack(
            [line1, np.tile(line1[-1], (len(line2) - len(line1), 1))])
    elif len(line2) < len(line1):
        line2 = np.vstack(
            [line2, np.tile(line2[-1], (len(line1) - len(line2), 1))])

    # Perform Procrustes analysis
    mtx1, mtx2, disparity = procrustes(line1, line2)

    return disparity


# Example usage
line1 = [(0, 0), (1, 2), (3, 3)]
line2 = [(2, 3), (3, 5), (5, 6)]  # Translated version of line1

disparity = compare_lines_procrustes(line1, line2)
print(f"The Procrustes disparity between the lines is: {disparity}")

#######


def compare_lines_dtw(line1, line2):
    # Compute the DTW distance
    distance, _ = fastdtw(line1, line2, dist=euclidean)
    return distance


# Example usage
line1 = [(0, 0), (1, 2), (3, 3)]
line2 = [(2, 3), (3, 5), (5, 6)]  # Translated version of line1

distance = compare_lines_dtw(line1, line2)
print(f"The DTW distance between the lines is: {distance}")


######

width, height, left_wrist_trajectory, right_wrist_trajectory, right_elbow_trajectory = track_shot()

# run_with_plot()
# display and saves the basketball trajectory
# load video
cap = cv2.VideoCapture('videos/nash_shot_clean.mp4')  # also: nash_cut.mp4
coordinates = track_basketball(
    cap, plot_save_file='plots/basketball_trajectory_7', csv_save_file='data/basketball_trajectory_7')

# print(f"Coordinates: {coordinates}")
# Plotting
fig, axs = plt.subplots(1, figsize=(8, 8))

# Plot left hand trajectory
x, y = zip(*left_wrist_trajectory)
axs.plot(x, y, marker='o', color="purple", markersize=2, label="Left wrist")

axs.set_title('Every Trajectory')
axs.set_aspect('equal')
axs.set_xlabel('X pixel')
axs.set_ylabel('Y pixel')
axs.set_xlim(0, width)
axs.set_ylim(height, 0)  # Flip y-axis to match image coordinates

# Plot right hand trajectory
x, y = zip(*right_wrist_trajectory)
axs.plot(x, y, marker='o', color="g", markersize=2, label="Right wrist")

# Plot right elbow trajectory
x2, y2 = zip(*right_elbow_trajectory)
axs.plot(x2, y2, marker='o', color="r", markersize=2, label="Right Elbow")

# Plot ball trajectory
x3, y3 = zip(*coordinates)
axs.plot(x3, y3, marker='o', color="b", markersize=2, label="Ball")

far_points, release_angle = find_release(
    40, right_wrist_trajectory, coordinates)

# Plot ball trajectory post release
x4, y4 = zip(*far_points)
axs.plot(x4, y4, marker='o', color="orange",
         markersize=2, label="Ball Released")

axs.plot(x4[:2], y4[:2], marker='o', color="yellow",
         markersize=1, label="First released points")

print(f"Deg: {release_angle:.2f}")

axs.legend()

plt.savefig('plots/full_trajectory_plot.png')  # Save the plot as an image
plt.close(fig)  # Close the plot to prevent displaying it


disparity_hands = compare_lines_procrustes(
    left_wrist_trajectory, right_wrist_trajectory)
print(f"The Procrustes disparity between the hands is: {disparity_hands}")
distance_hands = compare_lines_dtw(
    left_wrist_trajectory, right_wrist_trajectory)
print(f"The DTW distance between the hands is: {distance_hands}")
print(f"left: {left_wrist_trajectory}")
print(f"right: {right_wrist_trajectory}")
