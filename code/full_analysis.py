from track_motion import track_shot, find_release, compare_lines_dtw, compare_lines_procrustes, give_score
from score_basketballs import track_basketball
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Set backend to Agg


current_video = 'videos/nash_shot_clean.mp4'


# run_with_plot()
# display and saves the basketball trajectory
# load video
def score_shot(current_video, default_video='videos/nash_shot_clean.mp4'):

    # Get ball information for default video (from Steve Nash)
    cap1 = cv2.VideoCapture(default_video)  # also: nash_cut.mp4
    default_ball_coords = track_basketball(
        cap1, plot_save_file='plots/basketball_trajectory_7', csv_save_file='data/basketball_trajectory_7')

    # Get mediapipe info from default video (from Steve Nash)
    width, height, left_wrist_trajectory, right_wrist_trajectory, right_elbow_trajectory = track_shot()

    # Get ball information for test video (from our shots)
    cap2 = cv2.VideoCapture(current_video)  # also: nash_cut.mp4
    test_ball_coords = track_basketball(
        cap2, plot_save_file='plots/basketball_trajectory_7', csv_save_file='data/basketball_trajectory_7')

    # Get mediapipe info from test video (from our shots)
    test_width, test_height, test_left_wrist_traj, test_right_wrist_traj, test_right_elbow_traj = track_shot(
        current_video)

    ################### Plotting:##################
    # Plotting
    fig, axs = plt.subplots(1, figsize=(8, 8))

    # Plot left hand trajectory
    x, y = zip(*left_wrist_trajectory)
    axs.plot(x, y, marker='o', color="purple",
             markersize=2, label="Left wrist")

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
    x3, y3 = zip(*default_ball_coords)
    axs.plot(x3, y3, marker='o', color="b", markersize=2, label="Ball")

    # Get release angle and plot post release ball trajectory
    far_points, release_angle = find_release(
        40, right_wrist_trajectory, default_ball_coords)
    print(f"Deg: {release_angle:.2f}")

    # Plot ball trajectory post release
    x4, y4 = zip(*far_points)
    axs.plot(x4, y4, marker='o', color="orange",
             markersize=2, label="Ball Released")

    axs.plot(x4[:2], y4[:2], marker='o', color="yellow",
             markersize=1, label="First released points")

    axs.legend()

    plt.savefig('plots/full_trajectory_plot.png')  # Save the plot as an image
    plt.close(fig)  # Close the plot to prevent displaying it

    ############## Scoring ###############
    # Note: Release angle printed above

    fdtw_score = compare_lines_dtw(
        right_wrist_trajectory, test_right_wrist_traj)
    proc_score = compare_lines_procrustes(
        right_wrist_trajectory, test_right_wrist_traj)
    f_score, p_score = give_score(fdtw_score, proc_score)
    print(f"The FastDTW score out of 100 is: {f_score}")
    print(f"The Procrustes score out of 100 is: {p_score}")


if __name__ == '__main__':
    score_shot()
