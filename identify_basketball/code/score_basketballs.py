import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)

# read several testing images

frame1 = cv2.imread('../images/test/1.png', cv2.IMREAD_COLOR)
frame2 = cv2.imread('../images/test/2.png', cv2.IMREAD_COLOR)
frame3 = cv2.imread('../images/test/3.png', cv2.IMREAD_COLOR)
frame4 = cv2.imread('../images/test/4.png', cv2.IMREAD_COLOR)
frame5 = cv2.imread('../images/test/5.png', cv2.IMREAD_COLOR)
frame6 = cv2.imread('../images/test/nash.png', cv2.IMREAD_COLOR)

# score the contours based on their likelihood of being a basketball,
# and return the centroid of the most likely basketball
def find_basketball(frame):

   # convert the image to HSV
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Hue, Saturation, Value

   # define the color bounds for...

   # orange
   lower_orange = np.array([5, 100, 100])
   upper_orange = np.array([12, 255, 255])

   # red
   lower_red = np.array([172, 100, 100])
   upper_red = np.array([176, 255, 255])

   # dark red
   lower_dark_red = np.array([160,100,50])
   upper_dark_red = np.array([180,255,100])

   # Create a mask for the orange and red colors
   mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
   mask_red = cv2.inRange(hsv, lower_red, upper_red)
   mask_dark_red = cv2.inRange(hsv, lower_dark_red, upper_dark_red)

   # combine the masks
   mask = cv2.bitwise_or(mask_orange, mask_red)
   mask = cv2.bitwise_or(mask, mask_dark_red)

   # apply the mask
   masked_image = cv2.bitwise_and(frame, frame, mask=mask)







   # display the mask
   cv2.imshow("res", masked_image)
   cv2.waitKey(0)


find_basketball(frame1)
find_basketball(frame2)
find_basketball(frame3)
find_basketball(frame4)
find_basketball(frame5)
find_basketball(frame6)