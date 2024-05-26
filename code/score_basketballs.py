import cv2
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

# a default video of "None" will be the Nash shot
def track_basketball(cap, video, save_csv=True, save_plot=True, show_plot=True):
   # read the first frame
   ret, frame = cap.read()
   frame_shape = frame.shape

   # find the basketball in the first frame
   point_of_interest = find_basketball(frame, initial_frame=True, video=video)

   coordinates = []

   # find basketball for the whole video
   while cap.isOpened():
         
         ret, frame = cap.read()
   
         if not ret:
            break
   
         point_of_interest = find_basketball(frame, point_of_interest=point_of_interest, distance_weight=44, video=video)
         coordinates.append(point_of_interest)
   
   cap.release()

   coordinates = np.array(coordinates)

   # save the csv file
   if save_csv:
       np.savetxt('../data/' + video + '.csv', coordinates, delimiter=',')

   # make the plot
   plt.plot(coordinates[:, 0], frame_shape[0] - coordinates[:, 1])
   plt.title('Basketball Trajectory')

   # save png plot
   if save_plot:
       plt.savefig('../plots/' + video + '.png')

   # show plot
   if show_plot:
       plt.show()

   return coordinates

# score the contours based on their likelihood of being a basketball,
# and return the centroid of the most likely basketball
def find_basketball(frame, initial_frame=False, point_of_interest=None, square_weight=3, size_weight=10, distance_weight=10, video=None):
   print(video)



   # record the previous point of interest in case no new points are identified
   if point_of_interest != None:
      previous_point_of_interest = point_of_interest
   
   # manually define the point of interest from the first image
   if initial_frame == True and point_of_interest == None:
      if video == 'nash_shot_clean':
         point_of_interest = (frame.shape[1] / 4, frame.shape[0] / 2)
      elif video == 'srikanth_make':
         point_of_interest = (686, 801)
      elif video == 'srikanth_miss':
         point_of_interest = (686, 801)

   # draw point of interest
   cv2.circle(frame, (int(point_of_interest[0]), int(point_of_interest[1])), 5, (0, 0, 255), -1)

   # convert the image to HSV
   hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Hue, Saturation, Value

   # create a mask to segment out the basketball colors
   mask, masked_image = create_basketball_mask(hsv_image, frame, video) # masked image is it overlaid on the original image, mask is just the black and white image

   # remove the small objects (contours)
   cleaned_mask = remove_small_contours(mask, 175) # a larger threshold removes more noise

   # blur the image
   mask_blurred = cv2.blur(cleaned_mask, (3, 3))

   # find contours in the contour masked image
   contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   if len(contours) == 0:
      return previous_point_of_interest

   ########## For each contour, score it based on its likelihood of being a basketball ##########

   scores = [] # the total score for each contour

   square_scores = []
   size_scores = []
   distance_scores = []

   distance_check = []
  
   ### SQUARENESS SCORE ###
   for c in contours:
         x, y, w, h = cv2.boundingRect(c)

         # find how "square" it is
         squareness = min(w, h) / max(w, h) # how much bigger the big side is than the small side
         if squareness < 0.1: # if one side is much bigger than the other, it's probably not a basketball
             squareness = -100000

         square_scores.append(squareness) # the higher the score, the more likely it is a basketball. From 0 to 1

   ### SIZE SCORE ###

   for c in contours:
         size = cv2.contourArea(c)
         size_scores.append(size)

   # normalize the size scores
   size_scores = [s / max(size_scores) for s in size_scores]

   ### DISTANCE SCORE ###

   for c in contours:

      # find the center of each contour
      x, y, w, h = cv2.boundingRect(c)
      centroid_x = x + w / 2
      centroid_y = y + h / 2
      
      # this is here to prevent division by zero (if the centroid is the same as the point of interest, the distance is 0, which is not good for scoring)
      if (centroid_x, centroid_y) != point_of_interest:
         distance_from_point = 1 / np.linalg.norm(np.array([centroid_x, centroid_y]) - np.array(point_of_interest))
      else:
         distance_from_point = 0.000001

      distance_scores.append(distance_from_point)

   # normalize the distance scores
   distance_scores = [d / max(distance_scores) for d in distance_scores]

   ### DISTANCE CHECK ###    

   # for each contour, get the distance from the point of interest
   # this score allows us to "track" the basketball as its position changes
   for c in contours:
      x, y, w, h = cv2.boundingRect(c)
      centroid_x = x + w / 2
      centroid_y = y + h / 2
      distance = np.linalg.norm(np.array([centroid_x, centroid_y]) - np.array(point_of_interest))

      # basically disregarding the contour if it's too far away (unless all of them are >150 pixels away)
      if distance > 150:
         distance = -10000000
      else:
          distance = 0

      distance_check.append(distance)

   # apply weights, signifying the importance of each score
   square_scores = [s * square_weight for s in square_scores]
   size_scores = [s * size_weight for s in size_scores]
   distance_scores = [s * distance_weight for s in distance_scores]

   # add all the scores together, creating a single score for each contour
   scores = [square_scores[i] + size_scores[i] + distance_scores[i] + distance_check[i] for i in range(len(contours))]

   # find the contour with the highest score
   max_score_index = scores.index(max(scores))

   # this part of the code is for visualization purposes only
   for i, c in enumerate(contours):
         if i == max_score_index:
            x, y, w, h = cv2.boundingRect(c)
            cv2.putText(frame, str(round(max(scores), 2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            point_of_interest = (x + w / 2, y + h / 2)
         else:
               
            cv2.putText(frame, str(round(scores[i], 2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

   # if all of distance_check is less than 0, 
   # then the basketball's contour wasn't identified correctly, so
   # set the point of interest equal to the previous point of interest
   if all(i < 0 for i in distance_check) and initial_frame == False:
      point_of_interest = previous_point_of_interest

   cv2.imshow("Detected Basketball", frame)
   cv2.waitKey(0)


   return point_of_interest








        
# segments out the orange, red, and dark red colors of a typical basketball
def create_basketball_mask(hsv_image, frame, video):
    ### define the color bounds for... ###

   if video == 'nash_shot_clean':

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

   elif video == 'srikanth_make':       
      lower_orange = np.array([0, 120, 80])
      upper_orange = np.array([2, 235, 255])

      mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

   elif video == 'srikanth_miss':
       lower_orange = np.array([0, 120, 100])
       upper_orange = np.array([2, 255, 255])

       mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
       

       

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