#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:59:52 2019

@author: nisaak
"""

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import cv2
import numpy as np
import time

start_time = time.time()


path = "/home/nisaak/Documents/semester_thesis/export_test/day_cloudy2/left/frame0111.jpg"
image_right = cv2.imread(path, 1)


img_right = cv2.resize(image_right, dsize=(0,0), fx = 0.25, fy = 0.25)


right_labels = slic(img_right,300, 10, 10, 0)

sliced = mark_boundaries(img_right, right_labels)
sliced = cv2.resize(sliced, dsize = (0,0), fx = 4, fy = 4)

elapsed_time = time.time() - start_time

print(elapsed_time)
cv2.imshow('sliced', sliced)

cv2.waitKey()