from mediapipe1 import track_shot, find_release
from score_basketballs import track_basketball
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Set backend to Agg

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

far_points = find_release(40, right_wrist_trajectory, coordinates)

# Plot ball trajectory post release
x4, y4 = zip(*far_points)
axs.plot(x4, y4, marker='o', color="orange",
         markersize=2, label="Ball Released")


axs.legend()

plt.savefig('plots/full_trajectory_plot.png')  # Save the plot as an image
plt.close(fig)  # Close the plot to prevent displaying it