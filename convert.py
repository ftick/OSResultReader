import io
import os
from dotenv import load_dotenv

import json
import cv2
import numpy as np
import requests

load_dotenv()

API = os.getenv('API')
# FILE = "img/results1a"
FILE = "img/img1e"
JPG = FILE+".jpg"
TXT = FILE+".txt"

img = cv2.imread(JPG)
height, width, _ = img.shape

# Cutting image
# roi = img[100: height, 856: width]
roi = img[0: height, 0: width]

# Ocr
url_api = "https://api.ocr.space/parse/image"
_, compressedimage = cv2.imencode(".jpg", roi, [1, 90])
file_bytes = io.BytesIO(compressedimage)
result = requests.post(url_api,
  files = {JPG: file_bytes},
  data = {"apikey": API,
          "isTable": True,
          "language": "eng"})
result = result.content.decode()
result = json.loads(result)

parsed_results = result.get("ParsedResults")[0]
text_detected = parsed_results.get("ParsedText")

saveFile = open(TXT, "w")
saveFile.write(text_detected)