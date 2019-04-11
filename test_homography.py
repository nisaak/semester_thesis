#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:21:23 2019

@author: nisaak
"""

import cv2
import numpy as np
import sys
import os
import argparse
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--path_right', default ='/home/nisaak/Documents/semester_thesis/export_test/day_cloudy1/left/frame0000.jpg', type = str)
parser.add_argument('--path_left', default ='/home/nisaak/Documents/semester_thesis/export_test/day_cloudy1/right/frame0000.jpg', type = str)
#parser.add_argument('--verbose', default=False, type=bool)
args = parser.parse_args()


#def get_points(event, x, y, flags, param):
#    global lpnts
#    
#    if event == cv2.EVENT_LBUTTONDOWN:
#            lpnts = np.append(lpnts, np.array([[x,y]]), axis = 0)
            
path_right = args.path_right
path_left = args.path_left
img_right = cv2.imread(path_right)    
img_left = cv2.imread(path_left)

MIN_MATCH_COUNT = 10

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img_right, None)
kp2, des2 = orb.detectAndCompute(img_left, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print(M)

np.savetxt('homography_test_2', M)

matches = sorted(matches, key = lambda x:x.distance)

#good = []
#for m,n in matches:
#    if m.distance < 0.7*n.distance:
#        good.append(m)
#
#if len(good) > MIN_MATCH_COUNT:
#    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
#    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
#    
#    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#    matchesMask = mask.ravel().tolist()
#    
#    h,w = img_right.shape
#    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#    dst = cv2.perspectiveTransform(pts,M)
#
#    img2 = cv2.polylines(img_left,[np.int32(dst)],True,255,3, cv2.LINE_AA)

#else:
#    print('Not enough matches are found - %d/%d' % (len(good),MIN_MATCH_COUNT))
#    matchesMask = None
#    
#draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#            singlePointColor = None,
#            matchesMask = matchesMask, # draw only inliers
#            flags = 2)

#img3 = cv2.drawMatches(img_right,kp1,img_left,kp2,matches[:10], **draw_params)

plt.imshow(img3, 'gray'),plt.show()