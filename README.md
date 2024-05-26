# Computer Vision Final Project
Authors: Henry Buron, Srikanth Schelbert
Goal: Motion Tracking of Basketball Shots

## Dependencies
The following packages are required to use the code for this project
- FastDTW
- CV2
- Google MediaPipe
- Scipy Spatial
- Matplotlib
- Numpy

## Overview
This project aims to score a user's free-throw shot by running a copmarison against a ground truth example. In this case, that example is a free-throw from NBA Player Steve Nash. The project uses Google Mediapipe to track the motion of the players, and uses computer vision techniques like masking, object detection, and motion tracking to track the trajectory of the ball throughout the video.

The videos are analyzed, the code will produce plots and scores. One plot shows the trajectory of the hands, ball, and elbow while also showing the trajectory of the ball (separating the ball pre and post shot). The code will also return the release angle of the ball as well as a score out of 100 that acts as a simlarity score to the example throw. 