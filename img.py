import sys
from dotenv import load_dotenv

from PIL import Image, ImageOps

load_dotenv()
EQ = "eq.jpg"
OUT = "out.txt"

PATH = sys.argv[1]
img = Image.open(PATH)

# dims = img.size
width, height = img.size
cropped = img.crop((22*width/50,height/13,width,height*5/6))
grey = ImageOps.grayscale(cropped)
eq = ImageOps.equalize(grey)
eq.save(EQ)