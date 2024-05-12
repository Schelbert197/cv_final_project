import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)

# read the image (first frame)
first_frame = cv2.imread('../images/test/3.png', cv2.IMREAD_COLOR)

# convert the image to HSV
hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV) # Hue, Saturation, Value

# define the range of orange in HSV
lower_orange = np.array([3, 100, 100])
upper_orange = np.array([12, 255, 255])

# mask the image to get only orange colors
mask = cv2.inRange(hsv, lower_orange, upper_orange)

# close the mask
kernel = np.ones((5,5), np.uint8)
closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# bitwise and the mask and the original image
res = cv2.bitwise_and(first_frame, first_frame, mask=closed_mask)

# Convert to grayscale. 
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) 
  
# Blur using 3 * 3 kernel. 
gray_blurred = cv2.blur(gray, (3, 3)) 

# display the grayscale image
cv2.imshow('gray', gray_blurred)
cv2.waitKey(0)
  
# Apply Hough transform on the blurred image. 
detected_circles = cv2.HoughCircles(gray_blurred,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 30, minRadius = 1, maxRadius = 40) 
  
# Draw circles that are detected. 
if detected_circles is not None: 
  
    # Convert the circle parameters a, b and r to integers. 
    detected_circles = np.uint16(np.around(detected_circles)) 
  
    for pt in detected_circles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
  
        # Draw the circumference of the circle. 
        cv2.circle(first_frame, (a, b), r, (0, 255, 0), 2) 
  
        # Draw a small circle (of radius 1) to show the center. 
        cv2.circle(first_frame, (a, b), 1, (0, 0, 255), 3) 
        cv2.imshow("Detected Circle", first_frame) 
        cv2.waitKey(0) 