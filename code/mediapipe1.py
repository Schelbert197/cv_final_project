import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import cv2
import math
from scipy.spatial import distance
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg


# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def track_shot(video='videos/nash_shot_clean.mp4'):
    """
    Track the trajectory of specific body landmarks (right elbow, left wrist, and right wrist) 
    from a video using MediaPipe Holistic model.

    Parameters:
    -----------
    video : str, optional
        The path to the video file to be processed. Default is 'videos/nash_shot_clean.mp4'.

    Returns:
    --------
    width : int
        The width of the video frames.
    height : int
        The height of the video frames.
    left_wrist_trajectory : list of tuple
        A list of (x, y) coordinates representing the trajectory of the left wrist over time.
    right_wrist_trajectory : list of tuple
        A list of (x, y) coordinates representing the trajectory of the right wrist over time.
    right_elbow_trajectory : list of tuple
        A list of (x, y) coordinates representing the trajectory of the right elbow over time.

    Notes:
    ------
    The function opens a video file and processes it frame by frame to detect and track 
    specified landmarks. The trajectory of each landmark is stored in a list and returned 
    after the video is fully processed or the process is terminated by pressing 'q'.
    """
    # Load video
    cap = cv2.VideoCapture(video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:

        right_elbow_trajectory = []
        left_wrist_trajectory = []
        right_wrist_trajectory = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make detections
            results = holistic.process(image)

            # Draw landmarks on the face
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

            # Draw landmarks on the hands

            # Extract pixel-wise position of right elbow
            if results.pose_landmarks:
                right_elbow_pos = results.pose_landmarks.landmark[
                    mp_holistic.PoseLandmark.RIGHT_ELBOW.value]
                right_elbow_trajectory.append(
                    (int(right_elbow_pos.x * width), int(right_elbow_pos.y * height)))

                left_wrist_pos = results.pose_landmarks.landmark[
                    mp_holistic.PoseLandmark.LEFT_WRIST.value]
                left_wrist_trajectory.append(
                    (int(left_wrist_pos.x * width), int(left_wrist_pos.y * height)))

                right_wrist_pos = results.pose_landmarks.landmark[
                    mp_holistic.PoseLandmark.RIGHT_WRIST.value]
                right_wrist_trajectory.append(
                    (int(right_wrist_pos.x * width), int(right_wrist_pos.y * height)))

            # Draw landmarks on the pose
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Display output
            cv2.imshow('MediaPipe Holistic', cv2.cvtColor(
                image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return width, height, left_wrist_trajectory, right_wrist_trajectory, right_elbow_trajectory


def run_with_plot(video='videos/nash_shot_clean.mp4'):
    """
    Track body landmarks from a video and plot their trajectories.

    This function uses the `track_shot` function to track the trajectories of the left wrist, 
    right wrist, and right elbow from a video. It then plots these trajectories and saves 
    the plot as an image.

    Parameters:
    -----------
    video : str, optional
        The path to the video file to be processed. Default is 'videos/nash_shot_clean.mp4'.

    Returns:
    --------
    None

    Notes:
    ------
    The function first calls `track_shot` to obtain the trajectories of the specified landmarks. 
    It then plots these trajectories using Matplotlib, with the left wrist trajectory plotted 
    on the first subplot and both the right wrist and right elbow trajectories plotted on the 
    second subplot. The y-axis is flipped to match image coordinates. The resulting plot is 
    saved as 'plots/trajectory_plot.png'.
    """
    width, height, left_wrist_trajectory, right_wrist_trajectory, right_elbow_trajectory = track_shot(
        video)
    # Plotting
    fig, axs = plt.subplots(2, figsize=(8, 8))

    # Plot left hand trajectory
    x, y = zip(*left_wrist_trajectory)
    axs[0].plot(x, y, marker='o', markersize=2)

    axs[0].set_title('Left Hand Trajectory')
    axs[0].set_xlabel('X pixel')
    axs[0].set_ylabel('Y pixel')
    axs[0].set_xlim(0, width)
    axs[0].set_ylim(height, 0)  # Flip y-axis to match image coordinates

    # Plot right hand trajectory
    x, y = zip(*right_wrist_trajectory)
    axs[1].plot(x, y, marker='o', markersize=2)

    # Plot right elbow trajectory
    x2, y2 = zip(*right_elbow_trajectory)
    axs[1].plot(x2, y2, marker='o', color="r", markersize=2)

    axs[1].set_title('Right Hand Trajectory')
    axs[1].set_xlabel('X pixel')
    axs[1].set_ylabel('Y pixel')
    axs[1].set_xlim(0, width)
    axs[1].set_ylim(height, 0)  # Flip y-axis to match image coordinates

    plt.tight_layout()
    plt.savefig('plots/trajectory_plot.png')  # Save the plot as an image
    plt.close(fig)  # Close the plot to prevent displaying it


def find_release(threshold, wrist, ball):
    """
    Identify points in the 'ball' array that are farther than a specified
    threshold distance from the closest point in the 'wrist' array.

    This function first ensures that the 'wrist' array is no longer than
    the 'ball' array by removing excess elements from the end of 'wrist'.
    It then computes the Euclidean distances between each point in 'ball'
    and all points in 'wrist', finds the minimum distance for each point
    in 'ball', and identifies the points in 'ball' where this minimum
    distance exceeds the given threshold.

    Parameters:
    -----------
    threshold (float): The distance threshold for identifying far points.
    wrist (list of tuple): Array of (x, y) coordinates representing wrist points.
    ball (list of tuple): Array of (x, y) coordinates representing ball points.

    Returns:
    --------
    far_points (numpy.ndarray): Array of points in 'ball' that are farther
                   than the threshold distance from the nearest point in 'wrist'.
    release_angle (float): The release angle of the ball in degrees.
    """

    while len(wrist) > len(ball):
        wrist.pop()

    # Compute the distance from each point in ARR2 to each point in ARR1
    dists = distance.cdist(ball, wrist, 'euclidean')
    # Find the minimum distance to any point in ARR1 for each point in ARR2
    min_dists = np.min(dists, axis=1)
    # Find the indices where the distance is greater than the threshold
    indices = np.where(min_dists > threshold)[0]
    # Get the points in ARR2 that are farther than the threshold
    far_points = ball[indices]

    # Compute the release angle in degrees
    release_angle = math.degrees(
        math.atan2(-1*(far_points[1][1]-far_points[0][1]), (far_points[1][0]-far_points[0][0])))

    return far_points, release_angle
