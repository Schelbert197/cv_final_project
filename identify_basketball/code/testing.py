
import cv2 
import numpy as np 
  
# Read image. 
img = cv2.imread('../images/test/3.png', cv2.IMREAD_COLOR) 

# mask all colors except orange
# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of orange color in HSV

lower_orange = np.array([0, 100, 100])
upper_orange = np.array([10, 255, 255])

# Threshold the HSV image to get only orange colors
mask = cv2.inRange(hsv, lower_orange, upper_orange)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask=mask)

# cv2.imshow('img', img)

# cv2.imshow('mask', mask)

# cv2.imshow('res', res)

# cv2.waitKey(0)

# cv2.destroyAllWindows()

# close the res image

# binary = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

kernal = np.ones((5,5), np.uint8)

closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)

cv2.imshow('img', img)

cv2.imshow('mask', mask)

cv2.imshow('res', res)

cv2.imshow('closed', closed)

cv2.waitKey(0)





  
# # Convert to grayscale. 
# # gray = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY) 
  
# # Blur using 3 * 3 kernel. 
# gray_blurred = cv2.blur(closed, (3, 3)) 
  
# # Apply Hough transform on the blurred image. 
# detected_circles = cv2.HoughCircles(gray_blurred,  
#                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
#                param2 = 20, minRadius = 1, maxRadius = 90) 

# print(detected_circles)
# print(len(detected_circles[0]))
  
# # Draw circles that are detected. 
# if detected_circles is not None: 
  
#     # Convert the circle parameters a, b and r to integers. 
#     detected_circles = np.uint16(np.around(detected_circles)) 
  
#     for pt in detected_circles[0, :]: 
#         a, b, r = pt[0], pt[1], pt[2] 
  
#         # Draw the circumference of the circle. 
#         cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
  
#         # Draw a small circle (of radius 1) to show the center. 
#         cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
#         cv2.imshow("Detected Circle", img) 
#         cv2.waitKey(0) 