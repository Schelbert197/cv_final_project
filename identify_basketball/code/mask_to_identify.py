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
   closed_mask_1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

   # get rid of more noise
   kernel = np.ones((3,3), np.uint8)
   closed_mask = cv2.erode(closed_mask_1, kernel, iterations=1)

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
   
   centroid = 0
   object_coordinates = 0
   padding = 7
   
   # draw circles that are detected
   if detected_circles is not None:
      # may need to add a check for the number of circles detected
      # i.e. if greater than 1, use the largest circle

      detected_circles = np.uint16(np.around(detected_circles))
      for pt in detected_circles[0, :]:
         a, b, r = pt[0], pt[1], pt[2]

         # draw a box around the circle (basketball)
         top_left = (a-r-padding, b-r-padding)
         bottom_right = (a+r+padding, b+r+padding)
         cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

         cv2.imshow("Detected Circle", frame)
         cv2.waitKey(0)

         centroid = (a, b)
         object_coordinates = (a-r-padding, b-r-padding, a+r+padding, b+r+padding)

         return centroid, object_coordinates
      
   else:
      print("No circles detected, using contour...")

      # put a circle around teh largest contour
      contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      if len(contours) != 0:
         c = max(contours, key = cv2.contourArea)

         # draw a box around the circle (basketball)
         (x, y, w, h) = cv2.boundingRect(c)
         top_left = (x-padding, y-padding)
         bottom_right = (x+w+padding, y+h+padding)

         cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

         cv2.imshow("Detected Circle", frame)
         cv2.waitKey(0)

         centroid = (x+w//2, y+h//2)
         object_coordinates = (x-padding, y-padding, x+w+padding, y+h+padding)

         return centroid, object_coordinates


   






centroid, object_coordinates =  identify_basketball(frame1)
# centroid, object_coordinates = identify_basketball(frame2)
# centroid, object_coordinates = identify_basketball(frame3)

# cut the object from the frame
x1, y1, x2, y2 = object_coordinates
object = frame1[y1:y2, x1:x2]

cv2.imshow("Object", object)
cv2.waitKey(0)

print(centroid)
print(object_coordinates)