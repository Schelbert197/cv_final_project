from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import cv2


def create_pdf_report(filename=None, score=None, shooting_image=None, plots_image=None, launch_angle=None):

   IMG_SCALING_FACTOR = 4.3

   c = canvas.Canvas(filename, pagesize=letter)
   width, height = letter

   # set the pdf's title
   c.setTitle("Basketball Trajectory Report")

   # draw a title on the pdf
   c.setFont("Times-Roman", 24)
   c.drawCentredString(width / 2.0, height - 40, "Basketball Trajectory Report")

   # normalize the score
   normalized_score = score / 100.0

   # interpolate the color based on the score
   r = 1 - normalized_score
   g = normalized_score
   b = 0

   # draw a circle on the pdf
   c.setFillColorRGB(r, g, b)
   c.circle(width / 2.0, height - 88, 28, fill=1)

   # draw the score in the circle
   c.setFillColorRGB(0, 0, 0)
   c.setFont("Times-Roman", 28)
   c.drawCentredString(width / 2.0, height - 98, f"{score}")

   # get dimensions of the shooting image
   shooting_image_path = cv2.imread(shooting_image)
   
   # get image dimensions
   image_height, image_width, _ = shooting_image_path.shape

   image_width = image_width / IMG_SCALING_FACTOR
   image_height = image_height / IMG_SCALING_FACTOR

   # insert shooting image at release with superimposed trajectory ### TODO
   x = (width - image_width) / 2.0
   c.drawImage(shooting_image, x, y=410, width=image_width, height=image_height) # higher y value to move the image up

   # get dimensions of the plots image
   plots_image_path = cv2.imread(plots_image)

   # get image dimensions
   image_height_plots, image_width_plots, _ = plots_image_path.shape
   
   image_width_plots = image_width_plots / IMG_SCALING_FACTOR
   image_height_plots = image_height_plots / IMG_SCALING_FACTOR

   # insert plots image below shooting image
   x = (width - image_width_plots) / 2.0
   c.drawImage(plots_image, x, y=200, width=image_width_plots, height=image_height_plots)

   # include some text below the images
   c.setFont("Times-Roman", 12)

   # write text based on launch angle
   if launch_angle >= 45 and launch_angle <= 53:
      text = f"Great shot! Your shot's launch angle of {launch_angle} degrees is within the optimal range."
   elif launch_angle < 45:
      text = f"Your shot's launch angle of {launch_angle} degrees is too low. Try to aim higher next time."
   elif launch_angle > 53: 
      text = f"Your shot's launch angle of {launch_angle} degrees is too high. Try to aim lower next time."

   c.drawString(100, 150, text)



   

   # save the pdf
   c.save()




create_pdf_report("../reports/basketball_trajectory_report_1.pdf", 
                  score=92, 
                  launch_angle=41,
                  shooting_image="../images/nash_shooting.png",
                  plots_image="../images/trajectory_plot.png",)
