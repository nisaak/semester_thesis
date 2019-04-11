#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:58:04 2019

@author: nisaak
"""
import cv2
import numpy as np
  
def match(image_right, image_left):
    orb = cv2.ORB_create(500, scaleFactor=1.2,nlevels=8,edgeThreshold=31,firstLevel= 0,WTA_K= 2,scoreType=cv2.ORB_HARRIS_SCORE,patchSize=31,fastThreshold=20)
        
    kp1, des1 = orb.detectAndCompute(image_right, None)
    kp2, des2 = orb.detectAndCompute(image_left, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    matches = bf.match(des1, des2)
    
    kp_draw = cv2.drawKeypoints(image_right, kp1, None)
    cv2.imshow('kp', kp_draw)
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    

    F, mask_f = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 3.0, 0.99, cv2.FM_LMEDS)
    M, mask_m = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, 0.99)
    
    return M, F, src_pts, dst_pts