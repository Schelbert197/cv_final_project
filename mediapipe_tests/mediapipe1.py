import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg


# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Load video
# cap = cv2.VideoCapture('final_project/wr_ft_cut_covered.mp4')
# cap = cv2.VideoCapture('final_project/nash_miss_fullcut.mp4')
cap = cv2.VideoCapture('final_project/nash_slowed2.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize holistic model
with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:

    left_hand_trajectory = []
    right_hand_trajectory = []
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

        # # Draw landmarks on the hands
        # mp_drawing.draw_landmarks(
        #     image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(
        #     image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # # Extract pixel-wise positions of landmarks for each hand
        # if results.left_hand_landmarks:
        #     left_hand_trajectory.append([(int(landmark.x * image.shape[1]), int(
        #         landmark.y * image.shape[0])) for landmark in results.left_hand_landmarks.landmark][0])

        # if results.right_hand_landmarks:
        #     right_hand_trajectory.append([(int(landmark.x * image.shape[1]), int(
        #         landmark.y * image.shape[0])) for landmark in results.right_hand_landmarks.landmark][0])

        # # Extract pixel-wise positions of landmarks for each hand
        # if results.left_hand_landmarks:
        #     left_hand_trajectory.append([(int(landmark.x * width), int(landmark.y * height))
        #                                 for landmark in results.left_hand_landmarks.landmark][0])

        # if results.right_hand_landmarks:
        #     right_hand_trajectory.append([(int(landmark.x * width), int(landmark.y * height))
        #  for landmark in results.right_hand_landmarks.landmark][0])

        # print(type(results.pose_landmarks))
        # print('hi')
        # Extract pixel-wise position of right elbow
        if results.pose_landmarks:
            right_elbow_pos = results.pose_landmarks.landmark[
                mp_holistic.PoseLandmark.RIGHT_ELBOW.value]
            right_elbow_trajectory.append(
                (int(right_elbow_pos.x * width), int(right_elbow_pos.y * height)))

            left_wrist_pos = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST.value]
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
plt.savefig('trajectory_plot.png')  # Save the plot as an image
plt.close(fig)  # Close the plot to prevent displaying it
