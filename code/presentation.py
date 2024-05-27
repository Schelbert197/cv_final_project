import cv2
import numpy as np

# might want to make it so this can take in a list of lists of coordinates: [[(x1, y1), (x2, y2), ...], [(x1, y1), (x2, y2), ...], ...]
def superimpose_trajectory(image, coordinates, bgr_color=(0, 255, 0)):
      """
      Superimpose the basketball trajectory on the image.
   
      Parameters
      ----------
      image : numpy.ndarray
         The image to superimpose the trajectory on.
      coordinates : numpy.ndarray
         The coordinates of the basketball trajectory.
   
      Returns
      -------
      numpy.ndarray
         The image with the basketball trajectory superimposed.
      """

      # make a copy of the image
      image_with_trajectory = image.copy()
   
      # draw the trajectory on the image
      for i in range(1, len(coordinates)):
         cv2.line(image_with_trajectory, (int(coordinates[i - 1][0]), int(coordinates[i - 1][1])),
                  (int(coordinates[i][0]), int(coordinates[i][1])), bgr_color, 2)

      # show the image
      cv2.imshow('Basketball Trajectory Imposed', image_with_trajectory)
      cv2.waitKey(0)
   
      return image_with_trajectory