import io
import os
import sys
from dotenv import load_dotenv

import json
import cv2
import numpy as np
import requests

import sys
from PIL import Image, ImageOps

load_dotenv()

API = os.getenv('API')
EQ = "eq.jpg"

PATH = sys.argv[1]
img = Image.open(PATH)

# frame = img.resize((1920, 1080))

cropped = ImageOps.crop(img, 50)
cropped = img.crop((1050, 80, 1880, 890))
grey = ImageOps.grayscale(cropped)
eq = ImageOps.equalize(grey)
eq.save(EQ)

# a1 = eq.crop((40,0,296,410))
# a1.show()
# a2 = eq.crop((296,0,562,410))
# a2.show()
# a3 = eq.crop((562,0,830,410))
# a3.show()

img = cv2.imread(EQ)
height, width, _ = img.shape

# TXT = PATH+".txt"
TXT = "out.txt"

# Cutting image
# roi = img[100: height, 856: width]
roi = img[0: height, 0: width]

# Ocr
url_api = "https://api.ocr.space/parse/image"
_, compressedimage = cv2.imencode(".jpg", roi, [1, 90])
file_bytes = io.BytesIO(compressedimage)
result = requests.post(url_api,
  files = {EQ: file_bytes},
  data = {"apikey": API,
          "isTable": True,
          "language": "eng"})
result = result.content.decode()
result = json.loads(result)

parsed_results = result.get("ParsedResults")[0]
text_detected = parsed_results.get("ParsedText")
print(text_detected)

saveFile = open(TXT, "w")
saveFile.write(text_detected)

os.remove(EQ)