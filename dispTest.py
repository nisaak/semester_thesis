#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:04:16 2019

@author: nisaak

"""

import cv2
import numpy as np
import feat_match
import rectification_uncalib
import rectification_calib
import matplotlib

#specify path
path_left = '/home/nisaak/Documents/semester_thesis/export_test/day_cloudy1/left/frame0076.jpg'
path_right = '/home/nisaak/Documents/semester_thesis/export_test/day_cloudy1/right/frame0077.jpg'




#load images
img_left = cv2.imread(path_left,0)
img_left = cv2.resize(img_left, None, None,fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
BLACK = [0,0,0]
img_left = np.array(img_left, dtype= np.uint8)
cv2.imshow('left',img_left)
img_right = cv2.imread(path_right,0)
img_right = cv2.resize(img_right,None, None, fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)


img_right = np.array(img_right,dtype=np.uint8)

R_left = np.matrix('0.99934 0.000099 -0.036288; 0.000041 0.999993 0.003857; 0.036288 -0.003856 0.999334',np.float32)
R_right = np.matrix('0.998947 0.001982 -0.045833; -0.002159 0.999990 -0.003809; 0.045825 0.003904 0.998942',np.float32)
#M = np.matrix('1.053655553328680439e+00 -9.217688345630560554e-03 -1.312381196971389663e+01; \
#              9.701598541577670376e-03 1.019989885839541088e+00 -3.065585312977083632e+01; \
#              2.927936696573258345e-05 3.003417193742115971e-05 1.000000000000000000e+00')
#img_right_undis, img_left_undis = rectification_calib.rect_calib(img_right, img_left)

cam_mat_left = np.matrix('351.087640 0.000000 337.599686; 0.000000 351.004496 193.493580; 0.000000 0.000000 1.000000',np.float32)
dist_left = np.matrix('-0.166992 0.031731 0.002833 0.000323 0.000000',np.float32)
cam_mat_right = np.matrix('358.601746 0.000000 335.011174; 0.000000 358.378270 178.059030; 0.000000 0.000000 1.000000',np.float32)
dist_right  = np.matrix('-0.156551 0.018860 0.002804 -0.001476 0.000000',np.float32)

m1_left, m2_left = cv2.initUndistortRectifyMap(cam_mat_left, dist_left, R_left,None,(img_left.shape[1],img_left.shape[0]), cv2.CV_32FC1)
m1_right, m2_right = cv2.initUndistortRectifyMap(cam_mat_right, dist_right, R_right,None,(img_right.shape[1],img_left.shape[0]), cv2.CV_32FC1)

img_right_undis = cv2.remap(img_right, m1_right,m2_right, cv2.INTER_NEAREST)
img_left_undis = cv2.remap(img_left,m1_left,m2_left,cv2.INTER_NEAREST)

img_left_undis = cv2.copyMakeBorder(img_left_undis, 0,0,400,400, cv2.BORDER_CONSTANT, value = [255,255,255])

img_right_undis = cv2.copyMakeBorder(img_right_undis, 0,0,400,400, cv2.BORDER_CONSTANT, value = [255,255,255])


cv2.imshow('img_r', img_right_undis)
cv2.imshow('img_l',img_left_undis)
######
window_size = 9
minD = -21
numD = 16*13
blockS = 5
#####
stereo = cv2.StereoSGBM_create(minDisparity = minD,\
                               numDisparities=numD,\
                               blockSize=blockS,\
                               P1 = 8*window_size**2,\
                               P2 = 32*window_size**2,\
                               disp12MaxDiff = 12, \
                               preFilterCap = 63,\
                               uniquenessRatio = 7,\
                               speckleWindowSize = 0,\
                               speckleRange = 2)
"""
minDisparity = minD,\
                               numDisparities=numD,\
                               blockSize=blockS,P1 = 600,\
                               P2 = 2400,\
                               disp12MaxDiff = 10, \
                               preFilterCap = 4,\
                               uniquenessRatio = 1,\
                               speckleWindowSize = 150,\
                               speckleRange = 2)
"""

#M, _, _, _ = feat_match.match(img_right_undis, img_left_undis)
#
#img_left_warped = cv2.warpPerspective(img_left_rect, M, (img_left_rect.shape[1], img_left_rect.shape[0]))
#cv2.imshow('Left',img_left_warped)
#
#t_x = img_right.shape[1]/2.0
#t_y = img_right.shape[0]/2.0
#
#image_corr = np.array([[1,0,t_x],[0,1,t_y]],np.float32)
#img_right_undis = cv2.warpAffine(img_right_undis, image_corr, (img_right.shape[1],img_right.shape[0]))
#cv2.imshow('aff',img_right_undis)
#img_left_undis = cv2.warpAffine(img_left_undis, image_corr, (img_left.shape[1],img_left.shape[0]))
#
disparity = stereo.compute(img_left_undis, img_right_undis, None)
disparity = disparity[:,400:(disparity.shape[1]-400)]
disparity = cv2.normalize(disparity, None,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_HSV)
depth  = np.zeros_like(disparity)


cv2.imshow('disp', disparity)
cv2.waitKey()
cv2.destroyAllWindows()