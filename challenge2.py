#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
challenge2.py
Last Mod Nov 30 2020

@author: Conrad Borchers
"""

# Setup

import os

# os.chdir("C:/Users/conra/desktop/code_challenge")

from platform import python_version, platform
print(python_version()) # 3.7.0
print(platform())       # Win 10

import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
import layoutparser as lp
import cv2

# Converting all pdfs to jpg and moving them to /jpg

os.system("echo 'Collecting all pdf files in data/ and copying them to jpg/'")
os.system("mkdir jpg > nul 2> nul")
os.system("cp data/**/*.pdf jpg")

os.system("echo 'Converting all pdf files in jpg/ to jpg'")

files = ["jpg/" + f for f in os.listdir("jpg") if os.path.isfile("jpg/" + f) and f[-4:] == ".pdf"]

for f in files:
    pages = convert_from_path(f)
    page_count = 1
    for page in pages:
        file_name = f.replace(".pdf", "") + str(page_count) + ".jpg"
        page.save(file_name)
        page_count+=1

os.system("echo 'Deleting all pdf files in jpg/'")
os.system("rm jpg/*.pdf > nul 2> nul")

imgs = ["jpg/" + f for f in os.listdir("jpg") if os.path.isfile("jpg/" + f) and f[-4:] == ".jpg"]

# Get and apply model

test = imgs[0]

image = cv2.imread(test)
image = image[..., ::-1] 

# Detectron2LayoutModel function call throws windows path encoding error, to be continued ...

model = lp.Detectron2LayoutModel(
            config_path ='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', # In model catalog
            label_map   ={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}, # In model`label_map`
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
        )
model.detect(image)

lp.draw_box(image, layout, box_width=3)

