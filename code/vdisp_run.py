#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:11:06 2019

@author: nisaak
"""

import uvdisparity
import cv2
import numpy as np

path = "/home/nisaak/Documents/semester_thesis/code_test/disp_screenshot_30.04.2019_2/"

video_path = '/home/nisaak/Documents/semester_thesis/code_test/test_pictures/Disparitymap.avi'

cap = cv2.VideoCapture(video_path)


while True:
    
    ret, frame = cap.read()
    
    disparity_org = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('disp',disparity_org)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    disparity = np.copy(disparity_org)
#    disparity = cv2.imread(image_path,0)
    
    disparity = cv2.resize(disparity, dsize =(0,0),fx = 0.25, fy= 0.25)
    
    #background = disparity < 50
    #disparity[background] = 0
    
#    cv2.imshow('disparity', disparity)
    
    V_disp = uvdisparity.v_disp(disparity)
    
    
    
    U_disp = uvdisparity.u_disp(disparity)
    
    houghlines = uvdisparity.v_hough(V_disp, 1)

    u_houghlines = uvdisparity.u_hough(U_disp,1)
    
    mask = uvdisparity.mask(disparity, houghlines)
    
    mask = cv2.resize(mask, dsize =(0,0),fx = 4, fy= 4,interpolation = cv2.INTER_AREA)
    
    #find largest connected component of mask
    
    new_mask = np.zeros_like(mask)
    for val in np.unique(mask)[1:]:
        mask_for_labels = np.uint8(mask == val)
        labels, stats = cv2.connectedComponentsWithStats(mask_for_labels, 4)[1:3]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        new_mask[labels == largest_label] = val
        
    kernel = np.ones((1,1), np.uint8)
    kernel_close = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel)
#    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)
    new_mask = np.copy(closing)
    
    
#    cv2.imshow('U_disp', U_disp)
    cv2.imshow('new_mask', new_mask)
    
#    cv2.imshow('mask', mask)
#    
    cv2.imshow('hough_lines', houghlines)
#    
    cv2.imshow('V_disparity', V_disp)
    new_mask = cv2.resize(new_mask,dsize=(disparity_org.shape[1],disparity_org.shape[0]))
    print(disparity_org.shape)
    print(new_mask.shape)
    mask_on_img = cv2.addWeighted(disparity_org,1,new_mask,0.5, 5)
    cv2.imshow('maskonimg', mask_on_img)
    
    
cap.release()
cv2.destroyAllWindows()
