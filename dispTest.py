#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:04:16 2019

@author: nisaak

"""

import cv2
import numpy as np
import matplotlib.image
from uvdisparity import uv_disp
#specify path
path_left = '/home/nisaak/Documents/semester_thesis/export_test/day_cloudy1/left/frame0067.jpg'
path_right = '/home/nisaak/Documents/semester_thesis/export_test/day_cloudy1/right/frame0068.jpg'

video = 1




#load images

def load_images(path_right, path_left):
    
    #load images and save as grayscale
    
    img_left = cv2.imread(path_left,0)
    img_right = cv2.imread(path_right,0)
    
    #rescale images
    
    img_left = cv2.resize(img_left, None, None,fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA) 
    img_right = cv2.resize(img_right,None, None, fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)

    #convert to uint8
    img_left = np.array(img_left, dtype= np.uint8)
    img_right = np.array(img_right,dtype=np.uint8)
    
    
    return img_right, img_left


def rectify_and_remap(img_right, img_left):
    

    
    
    #rectification matrices
    
    R_left = np.array([[0.99934, 0.000099, -0.036288],\
                      [0.000041, 0.999993, 0.003857],\
                      [0.036288, -0.003856, 0.999334]],\
                       np.float32)
    
    R_right = np.array([[0.998947, 0.001982, -0.045833],\
                        [-0.002159, 0.999990, -0.003809],\
                        [0.045825, 0.003904, 0.998942]],\
                        np.float32)
    
    #camera matrices
    
    cam_mat_left = np.matrix([[351.087640, 0.000000, 337.599686],\
                             [0.000000, 351.004496, 193.493580],\
                             [0.000000, 0.000000, 1.000000]],\
                             np.float32)
    
    cam_mat_right = np.matrix([[358.601746, 0.000000, 335.011174],\
                              [0.000000, 358.378270, 178.059030],\
                              [0.000000, 0.000000, 1.000000]],\
                              np.float32)
    
    #distortion coefficients
    
    dist_left = np.array([-0.166992, 0.031731, 0.002833, 0.000323, 0.000000],np.float32)
    
    dist_right  = np.array([-0.156551, 0.018860, 0.002804, -0.001476, 0.000000],np.float32)
    
    
    
    """"
    these are the parameters from the official calibration file
    
    ///////
    cam_mat_left = np.matrix('350.409 0.00 338.233; 0.0 350.409 193.477; 0.0 0.0 1.0',np.float32)
    dist_left= np.matrix('-0.174318 0.0261121 0.002833 0.000323 0.0',np.float32)
    cam_mat_right = np.matrix('349.38 0.00 335.541;0.0 349.38 195.064; 0.0 0.0 1.0',np.float32)
    dist_right = np.matrix('-0.172731 0.0257849 0.002804 -0.001476 0.0',np.float32)
    
    M = np.matrix('1.053655553328680439e+00 -9.217688345630560554e-03 -1.312381196971389663e+01; \
                  9.701598541577670376e-03 1.019989885839541088e+00 -3.065585312977083632e+01; \
                  2.927936696573258345e-05 3.003417193742115971e-05 1.000000000000000000e+00')
    img_right_undis, img_left_undis = rectification_calib.rect_calib(img_right, img_left)
    
    ///////
    
    """
    
    #find remap matrices
    
    m1_left, m2_left = cv2.initUndistortRectifyMap(cam_mat_left, dist_left, R_left,None,(img_left.shape[1],img_left.shape[0]), cv2.CV_32FC1)
    
    m1_right, m2_right = cv2.initUndistortRectifyMap(cam_mat_right, dist_right, R_right,None,(img_left.shape[1],img_left.shape[0]), cv2.CV_32FC1)
    
    #remap
    
    img_right_undis = cv2.remap(img_right, m1_right,m2_right, cv2.INTER_CUBIC)
    
    img_left_undis = cv2.remap(img_left,m1_left,m2_left,cv2.INTER_CUBIC)
    
    #img_right_undis = cv2.normalize(img_right_undis, None, alpha=0, beta = 200, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    #img_left_undis = cv2.normalize(img_left_undis, None, alpha=0, beta = 200, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    
    return img_right_undis, img_left_undis








def disparity(img_right_undis, img_left_undis):
    
    
    #workaround for trunctuation bug, create border with sufficient size around image
    
    img_left_undis = cv2.copyMakeBorder(img_left_undis, 0,0,400,400, cv2.BORDER_CONSTANT, value = [255,255,255])

    img_right_undis = cv2.copyMakeBorder(img_right_undis, 0,0,400,400, cv2.BORDER_CONSTANT, value = [255,255,255])
    
    
######
    window_size = 11
    minD = -10
    numD = 16*5
    blockS = 9
    #####
    stereo = cv2.StereoSGBM_create(minDisparity = minD,\
                                   numDisparities=numD,\
                                   blockSize=blockS,\
                                   P1 = 8*window_size**2,\
                                   P2 = 32*window_size**2,\
                                   disp12MaxDiff = 12, \
                                   preFilterCap = 30,\
                                   uniquenessRatio = 1,\
                                   speckleWindowSize = 20,\
                                   speckleRange = 2)
    """
    ######
    window_size = 7
    minD = -20
    numD = 16*10
    blockS = 3
    #####
    stereo = cv2.StereoSGBM_create(minDisparity = minD,\
                                   numDisparities=numD,\
                                   blockSize=blockS,\
                                   P1 = 8*window_size**2,\
                                   P2 = 32*window_size**2,\
                                   disp12MaxDiff = 12, \
                                   preFilterCap = 30,\
                                   uniquenessRatio = 1,\
                                   speckleWindowSize = 50,\
                                   speckleRange = 1)
    """
    
    disparity = stereo.compute(img_left_undis, img_right_undis, None)
    disparity = disparity[:,400:(disparity.shape[1]-400)]
    disparity = cv2.normalize(disparity, None,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    disparity_coloured = cv2.applyColorMap(disparity, cv2.COLORMAP_HSV)
    
    return disparity, disparity_coloured

if video == 1:
#specify video loading
    video_left = '/home/nisaak/Documents/semester_thesis/export_test/dübendorf/left/left.avi'
    video_right = '/home/nisaak/Documents/semester_thesis/export_test/dübendorf/right/right.avi'
    
    
    names = [video_left,\
             video_right];
    window_titles = ['left', 'right']
    
    cap = [cv2.VideoCapture(i) for i in names]
    
    frames = [None] * len(names);
    gray = [None] * len(names);
    ret = [None] * len(names);
    
    while True:
    
        for i,c in enumerate(cap):
            if c is not None:
                ret[i], frames[i] = c.read();
                
        for i,f in enumerate(frames):
            if ret[i] is True:
                gray[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                cv2.imshow(window_titles[i], gray[i]);
      
                
                
        if cv2.waitKey(10) & 0xFF == ord('q'):
           break
       
        img_right = cv2.equalizeHist(np.copy(gray[1]))
        img_left = cv2.equalizeHist(np.copy(gray[0]))

        img_left = cv2.resize(img_left, None, None,fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA) 
        img_right = cv2.resize(img_right,None, None, fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)

        undis_right, undis_left = rectify_and_remap(img_right, img_left)
        
        disparity_img, _ = disparity(undis_right, undis_left)
        cv2.imshow('disp', disparity_img)
        uv_disp(disparity_img, img_right)
        
    
    for c in cap:
        if c is not None:
            c.release();
else:
    img_right, img_left = load_images(path_right, path_left)
    undis_right, undis_left = rectify_and_remap(img_right, img_left)
    disparity, _ = disparity(undis_right, undis_left)
    cv2.imshow('disp', disparity)   
    cv2.waitKey()     


cv2.waitKey(10)
cv2.destroyAllWindows()





