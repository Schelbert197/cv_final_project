from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_pdf_report(filename=None, score=None, image_1=None, image_2=None, launch_angle=None):

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
   c.circle(width / 2.0, height - 100, 35, fill=1)

   # draw a fraction in the circle
   c.setFillColorRGB(0, 0, 0)
   c.setFont("Times-Roman", 30)
   c.drawCentredString(width / 2.0, height - 110, f"{score}")

   # insert your image at release with superimposed trajectory
   image_width = 400
   x = (width - image_width) / 2.0
   c.drawImage(image_1, x, 330, width=image_width, height=300)

   # include some text below the image
   c.setFont("Times-Roman", 12)
   text = "Include some text here about the shot..."
   c.drawString(100, 250, text)

   # might want to include some of srikanth's plots, too?



   

   # save the pdf
   c.save()




create_pdf_report("../reports/basketball_trajectory_report_1.pdf", score=92, image_1="../images/nash_shooting.png")
