import os
import sys
from dotenv import load_dotenv

from PIL import Image, ImageOps
from paddleocr import PaddleOCR, draw_ocr

load_dotenv()

FONT_PATH = os.getenv('FONT_PATH')
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

outFile = open(OUT, "w")

ocr = PaddleOCR(use_angle_cls=True, lang="en", page_num=2)  # need to run only once to download and load model into memory
img_path = './{}'.format(EQ)
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        # print(line)
        outFile.write("{}: {}\n".format(line[0], line[1][0]))


# # draw result
import fitz
from PIL import Image
import cv2
import numpy as np
imgs = []
with fitz.open(img_path) as pdf:
    for pg in range(0, pdf.pageCount):
        page = pdf[pg]
        mat = fitz.Matrix(2, 2)
        pm = page.get_pixmap(matrix=mat, alpha=False)
        # if width or height > 2000 pixels, don't enlarge the image
        if pm.width > 2000 or pm.height > 2000:
            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        imgs.append(img)
for idx in range(len(result)):
    res = result[idx]
    image = imgs[idx]
    boxes = [line[0] for line in res]
    txts = [line[1][0] for line in res]
    scores = [line[1][1] for line in res]
    im_show = draw_ocr(image, boxes, txts, scores, font_path=FONT_PATH)
    im_show = Image.fromarray(im_show)
    im_show.save('result_3.jpg')#format(idx))