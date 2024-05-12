import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)

# read the image (first frame)
frame1 = cv2.imread('../images/test/1.png', cv2.IMREAD_COLOR)
frame2 = cv2.imread('../images/test/2.png', cv2.IMREAD_COLOR)
frame3 = cv2.imread('../images/test/3.png', cv2.IMREAD_COLOR)

def identify_basketball(frame):
   # convert the image to HSV
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Hue, Saturation, Value

   # define the range of orange in HSV
   lower_orange = np.array([3, 100, 100])
   upper_orange = np.array([12, 255, 255])

   # mask the image to get only orange colors
   mask = cv2.inRange(hsv, lower_orange, upper_orange)

   # close the mask
   kernel = np.ones((5,5), np.uint8)
   closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

   # bitwise and the mask and the original image
   res = cv2.bitwise_and(frame, frame, mask=closed_mask)

   # convert to grayscale
   gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

   # blur the image
   gray_blurred = cv2.blur(gray, (3, 3))

   # apply Hough transform on the blurred image
   detected_circles = cv2.HoughCircles(gray_blurred,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 30, minRadius = 1, maxRadius = 40)
   
   # draw circles that are detected
   if detected_circles is not None:
      print("dfjbsdkjfbsdkj")
      detected_circles = np.uint16(np.around(detected_circles))
      for pt in detected_circles[0, :]:
         a, b, r = pt[0], pt[1], pt[2]
         cv2.circle(frame, (a, b), r, (0, 255, 0), 2)
         cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)
         cv2.imshow("Detected Circle", frame)
         cv2.waitKey(0)
   else:
      print("No circles detected")
      # display the closed mask
      
      # cv2.imshow('closed', closed_mask)
      # cv2.waitKey(0)

      # put a circle around teh largest contour
      # contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


   






# identify_basketball(frame1)
identify_basketball(frame2)
# identify_basketball(frame3)