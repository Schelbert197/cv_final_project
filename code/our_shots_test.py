import cv2
import numpy as np


# load an image
image1 = cv2.imread('../images/test/test1.png')
image2 = cv2.imread('../images/test/test2.png')
image3 = cv2.imread('../images/test/test3.png')
image4 = cv2.imread('../images/test/test4.png')
image5 = cv2.imread('../images/test/test5.png')
image6 = cv2.imread('../images/test/test6.png')
image7 = cv2.imread('../images/test/test7.png')
image8 = cv2.imread('../images/test/test8.png')
image9 = cv2.imread('../images/test/test9.png')
image10 = cv2.imread('../images/test/test10.png')
image11 = cv2.imread('../images/test/test11.png')
image12 = cv2.imread('../images/test/test12.png')
image13 = cv2.imread('../images/test/test13.png')
image14 = cv2.imread('../images/test/test14.png')

images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11, image12, image13, image14]

for image in images:
      
      # convert image to hsv
      hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

      # define the range of the color orange in hsv
      lower_orange = np.array([0, 100, 100])
      upper_orange = np.array([15, 255, 255])

      # dark orange
      # lower_dark_orange = np.array([14, 75, 30])
      # upper_dark_orange = np.array([15, 100, 70])

      # define dark brown
      lower_dark_brown = np.array([0, 0, 0])
      upper_dark_brown = np.array([12, 12, 12])

      # create a mask for the color orange
      mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
      mask_brown = cv2.inRange(hsv_image, lower_dark_brown, upper_dark_brown)
      mask_dark_orange = cv2.inRange(hsv_image, lower_dark_orange, upper_dark_orange)

      # combine the masks
      mask = cv2.bitwise_or(mask_orange, mask_brown)
      mask = cv2.bitwise_or(mask, mask_dark_orange)

      # apply the mask to the image
      image = cv2.bitwise_and(image, image, mask=mask)






      # display the image
      cv2.imshow('image', image)
      cv2.waitKey(0)

