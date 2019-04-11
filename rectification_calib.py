#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:39:27 2019

@author: nisaak
"""

import numpy as np
import cv2


#specify camera parameters
#intrinsic parameters
#cam_mat_left = np.matrix('1401.64 0 1157.93; 0  1401.64 647.906;0 0 1',np.float64)
#cam_mat_right = np.matrix('1397.52 0 1107.16;0 1397.52 654.257;0 0 1',np.float64)

#distortion parameters
dis_coef_left = np.matrix('-0.174318 0.0261121 0 0', np.float64)
dis_coef_right = np.matrix('-0.172731 0.0257849 0 0',np.float64)

#extrinsic parameters
translation = np.matrix('-1200; 0; 0',np.float64)
R_rod = np.matrix('0.00497864;0.00958521;-0.00185401',np.float64)
R = np.zeros(shape=(3,3))
#R = np.identity(3)
cv2.Rodrigues(R_rod,R)


def rect_calib(img_right, img_left):
    R1, R2, P1, P2, Q = cv2.stereoRectify(cam_mat_right, dis_coef_right, cam_mat_left, dis_coef_left, \
                                          (img_right.shape[1], img_right.shape[0]),R,translation,alpha=1)[0:5]
    m1_right, m2_right = cv2.initUndistortRectifyMap(\
                                                     cam_mat_right, dis_coef_right,\
                                                     R1, \
                                                     None, \
                                                     (img_right.shape[1], img_right.shape[0]),\
                                                     cv2.CV_32FC1)
    
    m1_left, m2_left = cv2.initUndistortRectifyMap(cam_mat_left, dis_coef_left, R2, None, (img_right.shape[1], img_right.shape[0]),cv2.CV_32FC1)
    
    img_right_undis = (cv2.remap(img_right, m1_right, m2_right,0)).astype(np.uint8)
    
    
    img_left_undis = (cv2.remap(img_left, m1_left, m2_left,0)).astype(np.uint8)
    
    return img_right_undis, img_left_undis