from track_motion import track_shot, find_release, compare_lines_dtw, compare_lines_procrustes, give_score
from score_basketballs import track_basketball
from create_pdf_report import write_pdf_report
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Set backend to Agg


current_video = 'videos/nash_shot_clean.mp4'


def choose_video():
    """Prompts the user to select a video from the list given."""
    # Define a dictionary to map selections to video titles
    video_dict = {
        1: 'srikanth_make',
        2: 'srikanth_miss',
        3: 'henry_make',
        4: 'henry_miss'
    }
    skill_levels = ['p', 'i', 'b']

    # Run until user selects a valid video
    while True:
        try:
            selection = int(input("Please select a number corresponding to a video:\n \
                          1: srikanth_make\n \
                          2: srikanth_miss\n \
                          3: henry_make\n \
                          4: henry_miss\n"))
            if selection in video_dict:
                video = video_dict[selection]
                break
            else:
                print("Please select a number corresponding to a valid video")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 4.")

    while True:
        skill_level = input("Please select your skill level with p, i, or b:\n \
                        p: Pro\n \
                        i: Intermediate\n \
                        b: Beginner\n")
        if skill_level in skill_levels:
            break
        else:
            print("Please select a skill level with p, i, or b")

    print(f"Selected video: {video}")

    return video, skill_level


def score_shot(current_video, default_video='nash_shot', skill='i'):
    """Main function to run full analysis"""
    if skill == 'p':
        skill_factor = 1.0
    elif skill == 'i':
        skill_factor = 0.25
    else:
        skill_factor = 0.1

    # Get mediapipe info from default video (from Steve Nash)
    width, height, left_wrist_trajectory, right_wrist_trajectory, right_elbow_trajectory = track_shot(
        'videos/vids/' + default_video + '.mp4')

    # Get ball information for default video (from Steve Nash)
    default_ball_coords = track_basketball(
        video=default_video,
        save_csv=False,
        save_plot=False,
        show_plot=False)

    # Get mediapipe info from test video (from our shots)
    test_width, test_height, test_left_wrist_traj, test_right_wrist_traj, test_right_elbow_traj = track_shot(
        'videos/vids/' + current_video + '.mp4')

    # Get ball information for test video (from our shots)

    test_ball_coords = track_basketball(
        video=current_video,
        save_csv=False,
        save_plot=False,
        show_plot=False
    )
    if default_ball_coords.any() and test_ball_coords.any():
        print("Yeet got the goods.")
    ################### Plotting:##################
    # Plotting
    print("starting plots")
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot left hand trajectory
    x, y = zip(*left_wrist_trajectory)
    axs[0].plot(x, y, marker='o', color="purple",
                markersize=2, label="Left wrist")

    axs[0].set_title('Every Trajectory (Steve Nash)')
    axs[0].set_aspect('equal')
    axs[0].set_xlabel('X pixel')
    axs[0].set_ylabel('Y pixel')
    axs[0].set_xlim(0, width)
    # Flip y-axis to match image coordinates
    axs[0].set_ylim(height, 0)

    # Plot right hand trajectory
    x, y = zip(*right_wrist_trajectory)
    axs[0].plot(x, y, marker='o', color="g",
                markersize=2, label="Right wrist")

    # Plot right elbow trajectory
    x2, y2 = zip(*right_elbow_trajectory)
    axs[0].plot(x2, y2, marker='o', color="r",
                markersize=2, label="Right Elbow")

    # Plot ball trajectory
    x3, y3 = zip(*default_ball_coords)
    axs[0].plot(x3, y3, marker='o', color="b", markersize=2, label="Ball")

    # Get release angle and plot post release ball trajectory
    far_points, release_angle = find_release(
        40, right_wrist_trajectory, default_ball_coords)
    print(f"Deg: {release_angle:.2f}")

    # Plot ball trajectory post release
    x4, y4 = zip(*far_points)
    axs[0].plot(x4, y4, marker='o', color="orange",
                markersize=2, label="Ball Released")

    axs[0].plot(x4[:2], y4[:2], marker='o', color="yellow",
                markersize=1, label="First released points")

    axs[0].legend()

    # Plot test data
    # Plot left hand trajectory
    test_x, test_y = zip(*test_left_wrist_traj)
    axs[1].plot(test_x, test_y, marker='o', color="purple",
                markersize=2, label="Left wrist")

    axs[1].set_title('Every Trajectory (' + current_video + ')')
    axs[1].set_aspect('equal')
    axs[1].set_xlabel('X pixel')
    axs[1].set_ylabel('Y pixel')
    axs[1].set_xlim(0, test_width)
    axs[1].set_ylim(test_height, 0)  # Flip y-axis to match image coordinates

    # Plot right hand trajectory
    test_x, test_y = zip(*test_right_wrist_traj)
    axs[1].plot(test_x, test_y, marker='o', color="g",
                markersize=2, label="Right wrist")

    # Plot right elbow trajectory
    test_x2, test_y2 = zip(*test_right_elbow_traj)
    axs[1].plot(test_x2, test_y2, marker='o', color="r",
                markersize=2, label="Right Elbow")

    # Plot ball trajectory
    test_x3, test_y3 = zip(*test_ball_coords)
    axs[1].plot(test_x3, test_y3, marker='o',
                color="b", markersize=2, label="Ball")

    # Get release angle and plot post release ball trajectory
    test_far_points, test_release_angle = find_release(
        50, test_left_wrist_traj, test_ball_coords)
    print(f"Deg: {test_release_angle:.2f}")

    # Plot ball trajectory post release
    test_x4, test_y4 = zip(*test_far_points)
    axs[1].plot(test_x4, test_y4, marker='o', color="orange",
                markersize=2, label="Ball Released")

    axs[1].plot(test_x4[:2], test_y4[:2], marker='o', color="yellow",
                markersize=1, label="First released points")

    axs[1].legend()

    # Save the plot as an image
    plt.savefig(f'plots/final_output_{current_video}.png')
    plt.close(fig)  # Close the plot to prevent displaying it

    ############## Scoring ###############
    # Note: Release angle printed above

    fdtw_score = compare_lines_dtw(
        right_wrist_trajectory, test_left_wrist_traj)
    proc_score = compare_lines_procrustes(
        right_wrist_trajectory, test_left_wrist_traj)
    f_score, p_score = give_score(fdtw_score, proc_score, skill_factor)
    overall = (f_score + p_score)/2
    print(f"The FastDTW score out of 100 is: {f_score}")
    print(f"The Procrustes score out of 100 is: {p_score}")
    print(f"Overall Score: {overall}")

    fdtw_score_ball = compare_lines_dtw(
        default_ball_coords, test_ball_coords)
    proc_score_ball = compare_lines_procrustes(
        default_ball_coords, test_ball_coords)
    f_score_ball, p_score_ball = give_score(
        fdtw_score_ball, proc_score_ball, skill_factor)
    overall_ball = (f_score_ball + p_score_ball)/2
    print(f"The FastDTW score out of 100 is: {f_score_ball}")
    print(f"The Procrustes score out of 100 is: {p_score_ball}")
    print(f"Overall Score: {overall_ball}")

    fdtw_score_b = compare_lines_dtw(
        far_points, test_far_points)
    proc_score_b = compare_lines_procrustes(
        far_points, test_far_points)
    f_score_b, p_score_b = give_score(fdtw_score_b, proc_score_b, skill_factor)
    overall_b = (f_score_b + p_score_b)/2
    print(f"The FastDTW score out of 100 is: {f_score_b}")
    print(f"The Procrustes score out of 100 is: {p_score_b}")
    print(f"Overall Score: {overall_b}")

    weights = np.array([0.25, 0.5, 0.25])
    overalls = np.array([overall, overall_ball, overall_b])
    final_score = np.dot(weights, overalls)
    all_scores = [f_score,
                  p_score,
                  overall,
                  f_score_ball,
                  p_score_ball,
                  overall_ball,
                  f_score_b,
                  p_score_b,
                  overall_b,
                  final_score]
    write_pdf_report('reports/basketball_trajectory_report_' + current_video + '.pdf',
                     score=int(final_score),
                     launch_angle=int(test_release_angle),
                     shooting_image='images/' + current_video + '_traj.png',
                     plots_image=f'plots/final_output_{current_video}.png'
                     )
    return all_scores


if __name__ == '__main__':
    chosen_video, skill_level = choose_video()
    scores = score_shot(current_video=chosen_video, skill=skill_level)
    print(f"All scores: {scores}")
