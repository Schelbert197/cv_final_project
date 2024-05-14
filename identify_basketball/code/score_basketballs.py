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
def find_basketball(frame, initial_frame=False, point_of_interest=None, square_weight=3, size_weight=1, distance_weight=10):

   # convert the image to HSV
   hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Hue, Saturation, Value

   # create a mask to segment out the basketball colors
   mask, masked_image = create_basketball_mask(hsv_image, frame) # masked image is it overlaid on the original image, mask is just the black and white image

   # remove the small objects (contours)
   cleaned_mask = remove_small_contours(mask, 175) # a larger threshold removes more noise

   # blur the image
   mask_blurred = cv2.blur(cleaned_mask, (3, 3))

   # # find contours in the contour masked image
   contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   ########## For each contour, score it based on its likelihood of being a basketball ##########

   scores = [] # the total score for each contour

   square_scores = []

   size_scores = []

   distance_scores = []
  
   ### SQUARENESS SCORE ###
   for c in contours:
         x, y, w, h = cv2.boundingRect(c)

         # find how "square" it is
         small = min(w, h)
         big = max(w, h)
         squareness = small / big # how much bigger the bigger side is than the smaller side
         square_scores.append(squareness) # the higher the score, the more likely it is a basketball. From 0 to 1

   ### SIZE SCORE ###

   for c in contours:
         size = cv2.contourArea(c)
         size_scores.append(size)

   ### DISTANCE SCORE ###

   for c in contours:
      x, y, w, h = cv2.boundingRect(c)
      centroid_x = x + w / 2
      centroid_y = y + h / 2

      # distance from the point of interest
      if initial_frame:
         point_of_interest = (frame.shape[1] / 4, frame.shape[0] / 2) # center of the frame

         distance_from_point = distance(centroid_x, centroid_y, point_of_interest[0], point_of_interest[1])
      
      else:
         distance_from_point = distance(centroid_x, centroid_y, point_of_interest[0], point_of_interest[1])

      distance_from_point = distance_from_point ** -1 # make is so larger val = closer to the point of interest (i.e. good)
      distance_scores.append(distance_from_point)

   # normalize size and distance
   size_scores = [s / max(size_scores) for s in size_scores]
   distance_scores = [d / max(distance_scores) for d in distance_scores]

   # apply weights
   square_scores = [s * square_weight for s in square_scores]
   size_scores = [s * size_weight for s in size_scores]
   distance_scores = [s * distance_weight for s in distance_scores]

   # add all the scores together
   scores = [square_scores[i] + size_scores[i] + distance_scores[i] for i in range(len(contours))]


   ### make sure each contour is close enough ###
   if initial_frame == False:
       for i, c in enumerate(contours):
           x, y, w, h = cv2.boundingRect(c)
           centroid_x = x + w / 2
           centroid_y = y + h / 2

           if distance(centroid_x, centroid_y, point_of_interest[0], point_of_interest[1]) > 500:
               scores[i] = 0




   # find the contour with the highest score
   max_score = max(scores)
   max_score_index = scores.index(max_score)

   for i, c in enumerate(contours):
         if i == max_score_index:
               x, y, w, h = cv2.boundingRect(c)
               # print the score on the rectangle
               cv2.putText(frame, str(round(max_score, 2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


               cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
         else:
               
               pass
         # print the score on the rectangle
               cv2.putText(frame, str(round(scores[i], 2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
               x, y, w, h = cv2.boundingRect(c)
               cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


   # draw point of interest
   cv2.circle(frame, (int(point_of_interest[0]), int(point_of_interest[1])), 5, (0, 0, 255), -1)

   cv2.imshow("Detected Basketball", frame)
   cv2.waitKey(0)

   point_of_interest = (x + w / 2, y + h / 2)


   return point_of_interest








        
# segments out the orange, red, and dark red colors of a typical basketball
def create_basketball_mask(hsv_image, frame):
    ### define the color bounds for... ###

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
   mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
   mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
   mask_dark_red = cv2.inRange(hsv_image, lower_dark_red, upper_dark_red)

   # combine the masks
   mask = cv2.bitwise_or(mask_orange, mask_red)
   mask = cv2.bitwise_or(mask, mask_dark_red)

   # apply the mask
   masked_image = cv2.bitwise_and(frame, frame, mask=mask)

   # mask is the black and white image, masked_image is the original image with the mask applied

   return mask, masked_image

def remove_small_contours(mask, threshold):
   ### remove the small contours ###
   contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   contours = [c for c in contours if cv2.contourArea(c) > threshold]

   # create the image without the contours
   cleaned_mask = np.zeros_like(mask) # the black and white image

   for c in contours:
         cv2.drawContours(cleaned_mask, [c], -1, 255, -1)

   return cleaned_mask

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    


      


# find_basketball(frame1, initial_frame=True)
# find_basketball(frame2, initial_frame=True)
# find_basketball(frame3, initial_frame=True)
# find_basketball(frame4, initial_frame=True)
# find_basketball(frame5, initial_frame=True)
# find_basketball(frame6, initial_frame=True)