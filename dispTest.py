#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:04:16 2019

@author: nisaak

"""

import cv2
import numpy as np
import matplotlib.image
import uvdisparity
#specify path

#path = "/home/nisaak/Documents/semester_thesis/datasets/malaga-urban-dataset-extract-01/Images"
#path_left = path + "/left/img_CAMERA1_1261228749.918590_left.jpg"
#path_right = path + "/right/img_CAMERA1_1261228749.918590_right.jpg"

path_left = "/home/nisaak/Documents/semester_thesis/export_test/day_cloudy2/left/frame0111.jpg"
path_right = "/home/nisaak/Documents/semester_thesis/export_test/day_cloudy2/right/frame0111.jpg"

video = 1



#load images

def load_images(path_right, path_left, grayscale):
    
    #load images and save as grayscale
    
    img_left = cv2.imread(path_left, grayscale)

    img_right = cv2.imread(path_right, grayscale)
    
    if img_right is None:
        print('right image could not be read')
    if img_left is None:
        print('left image could not be read')
    
    return img_right, img_left

def crop(img_left, img_right):
    
    img_left_cropped = np.copy(img_left[140:590, :])
    img_right_cropped = np.copy(img_right[140:590, :])

    return img_left_cropped, img_right_cropped


def rectify_and_remap(img_right, img_left):

    height, width = img_right.shape

    #left
    
    cam_mat_left = np.matrix([[701.027680, 0.000000, 639.252473],
                              [0.000000, 700.144686, 371.672457],
                              [0.000000, 0.000000, 1.000000]], np.float32)
    
    dist_left = np.matrix([-0.171607, 0.025746, 0.001324, -0.000930, 0.000000], np.float32)
    
    R_left = np.matrix([[0.999981, -0.002404, 0.005590],
                        [0.002341, 0.999934, 0.011208],
                        [-0.005617, -0.011194, 0.999922]],np.float32)
    
    P_left = np.matrix([[693.172274, 0.000000, 628.266212, 0.000000],
                        [0.000000, 693.172274, 349.947464, 0.000000],
                        [0.000000, 0.000000, 1.000000, 0.000000]], np.float32)
    
    #right
    
    cam_mat_right  = np.matrix([[700.187722, 0.000000, 641.248870],
                                [0.000000, 699.757218, 330.073207],
                                [0.000000, 0.000000, 1.000000]], np.float32)
    
    dist_right = np.matrix([-0.174720, 0.025240, -0.000252, -0.000938, 0.000000],np.float32)
    
    R_right = np.matrix([[0.999986, -0.001631, 0.005038],
                         [0.001687, 0.999936, -0.011197],
                         [-0.005020, 0.011205, 0.999925]],np.float32)
    
    P_right = np.matrix([[693.172274, 0.000000, 628.266212, -83.243404],
                         [0.000000, 693.172274, 349.947464, 0.000000],
                         [0.000000, 0.000000, 1.000000, 0.000000]],np.float32)

    #find remap matrices
    
    m1_left, m2_left = cv2.initUndistortRectifyMap(cam_mat_left, dist_left, R_left, P_left, (width, height), cv2.CV_32FC1)
    
    m1_right, m2_right = cv2.initUndistortRectifyMap(cam_mat_right, dist_right, R_right, P_right, (width,height), cv2.CV_32FC1)
    
    #remap
    
    img_right_undis = cv2.remap(img_right, m1_right,m2_right, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    
    img_left_undis = cv2.remap(img_left,m1_left,m2_left,cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    

    return img_right_undis, img_left_undis







def disparity(img_right_undis, img_left_undis, method):
    
    img_left_undis, img_right_undis = crop(img_left_undis, img_right_undis)
    
    img_right_undis = cv2.GaussianBlur(img_right_undis, (7,7),1)
    img_left_undis = cv2.GaussianBlur(img_left_undis, (7,7),1)
#    sharp_right = cv2.addWeighted(img_right_undis, 1, filt_right, -0.7, 1)
#    sharp_left = cv2.addWeighted(img_left_undis, 1, filt_left, -0.7, 1)
#    cv2.imshow('filt', filt_right)
#    cv2.imshow('sharp', sharp_right)

    left_for_matcher = cv2.resize(np.copy(img_left_undis), dsize=(0,0), fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR_EXACT)
    right_for_matcher = cv2.resize(np.copy(img_right_undis), dsize=(0,0), fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR_EXACT)
    
    cv2.imshow('matcher', right_for_matcher)

    if method == 1: #sgbm
        window_size = 13
        minD = -12
        numD = 3*16
        blockS = 5
        #####
     
        matcher_left = cv2.StereoSGBM_create(minDisparity = minD,\
                                       numDisparities=numD,\
                                       blockSize=blockS,\
                                       P1 = 8*window_size**2,\
                                       P2 = 32*window_size**2,\
                                       disp12MaxDiff = -1, \
                                       preFilterCap = 63,\
                                       uniquenessRatio = 0,\
                                       speckleWindowSize = 0,\
                                       speckleRange = 20,
                                       mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    else: #bm
        matcher_left = cv2.StereoBM.create(3*16,7)
        minD = 0
        matcher_left.setMinDisparity(minD)
        matcher_left.setDisp12MaxDiff(1)
        matcher_left.setTextureThreshold(0)
        matcher_left.setSpeckleWindowSize(20)
        matcher_left.setSpeckleRange(32)
        matcher_left.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        matcher_left.setUniquenessRatio(2)
        matcher_left.setPreFilterCap(5)
        matcher_left.setPreFilterSize(15)
    
    #post processing for disparity map
    #filter parameters
    LAMBDA = 2000
    SIGMA = 1.2
    
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left = matcher_left)
    matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)

    wls.setLambda(LAMBDA)
    wls.setSigmaColor(SIGMA)
    
    
    right_for_filter = np.copy(img_right_undis)
        
    left_for_filter = np.copy(img_left_undis)
    
    
    disparity_right = matcher_right.compute(right_for_matcher, left_for_matcher, None)
    
    disparity_left = matcher_left.compute(left_for_matcher, right_for_matcher, None)

    disparity_left_wls = np.empty_like(disparity_left)
    
        
    wls.filter(disparity_left, left_for_filter, disparity_left_wls, disparity_right, right_view = right_for_filter)
    
    
    conf_map  = wls.getConfidenceMap()
    cv2.normalize(conf_map, conf_map,0, 255, cv2.NORM_MINMAX)
    conf_map = conf_map.astype(np.uint8)
    disparity_left=cv2.resize(disparity_left, dsize=(1280,450))
    conf_mask = conf_map < 10
    disparity_left[conf_mask] = 0
    
#    conf_disp = np.multiply(disparity_left,
#    cv2.normalize(conf_map,conf_map, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    cv2.imshow('conf-map', conf_map)
    
    return disparity_left, disparity_left_wls

if video == 1:
#specify video loading
    video_left = '/home/nisaak/Documents/semester_thesis/export_test/day_cloudy2/left/output.ogv'
    video_right = '/home/nisaak/Documents/semester_thesis/export_test/day_cloudy2/right/output.ogv'

    
    
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
#                gray[i] = f
                cv2.imshow(window_titles[i], gray[i]);
      
                
                
        if cv2.waitKey(5) & 0xFF == ord('q'):
           break

        img_right = np.copy(gray[1])
        img_left = np.copy(gray[0])
        
        left_rect, right_rect = rectify_and_remap(img_right, img_left)

        
        disparity_left, disparity_left_wls = disparity(right_rect, left_rect,1)
        disparity_left = cv2.normalize(disparity_left, None, 0, 255, cv2.NORM_MINMAX)
        disparity_left = disparity_left.astype(np.uint8)

        
        
        #now calculate uv_disparity
        V_disp = uvdisparity.v_disp(disparity_left)
        cv2.imshow('V_disp', V_disp)
        cdst = uvdisparity.v_hough(V_disp, 1)
        
        cv2.imshow('cdst',cdst)
#        
        disparity_left = cv2.resize(disparity_left, dsize =(0,0),fx = 0.25, fy= 0.25)

        
        mask = uvdisparity.mask(disparity_left, cdst)
        
        mask = cv2.resize(mask, dsize =(0,0),fx = 4, fy= 4)

        cv2.imshow('mask', mask)
        
        #change dtype for visualisation
        DEPTH_VISUALIZATION_SCALE = 255

        disparity_left = cv2.resize(disparity_left, dsize=(0,0),fx=4,fy=4)
        cv2.imshow('disp', disparity_left)
        cv2.imshow('disp_wls', disparity_left_wls/DEPTH_VISUALIZATION_SCALE)
        


        
    
    for c in cap:
        if c is not None:
            c.release();



else:
    img_right, img_left = load_images(path_right, path_left, 0)
    
    
    height, width = img_right.shape
    
    
    cv2.imshow('right', img_right)
    cv2.imshow('left', img_left)
    left_rect, right_rect = rectify_and_remap(img_right, img_left)

    
    disparity_left, disparity_left_wls = disparity(right_rect, left_rect,1)
    
    
    disparity_left = cv2.normalize(disparity_left, None, 0, 255, cv2.NORM_MINMAX)
    disparity_left = disparity_left.astype(np.uint8)
    cv2.imshow('disp', disparity_left)
    cv2.imshow('disp_wls', disparity_left_wls)


    
    #now calculate uv_disparity
    V_disp = uvdisparity.v_disp(disparity_left)
    cv2.imshow('V_disp', V_disp)
    cdst = uvdisparity.v_hough(V_disp, 1)
    
    cv2.imshow('cdst',cdst)

    
    disparity_left = cv2.resize(disparity_left, dsize =(0,0),fx = 0.25, fy= 0.25,interpolation =  cv2.INTER_AREA)

    
    mask = uvdisparity.mask(disparity_left, cdst)
    
    mask = cv2.resize(mask, dsize =(0,0),fx = 4, fy= 4)

    cv2.imshow('mask', mask)
    
    #change dtype for visualisation
#    DEPTH_VISUALIZATION_SCALE = 255
#        disparity_left = cv2.normalize(disparity_left, None, 0, 255, cv2.NORM_MINMAX, dtype= cv2.CV_8U)
#        disparity_left_wls = cv2.normalize(disparity_left_wls, None,0,255, cv2.NORM_MINMAX,dtype= cv2.CV_8U)

    cv2.waitKey()

cv2.destroyAllWindows()





