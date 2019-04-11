#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:10:51 2019

@author: nisaak
"""

import cv2
import feat_match

def rect(src_pts, dst_pts, img_right, img_left, F_mat):
    (w, h) = img_right.shape[1], img_right.shape[0]
    print((h,w))
    _, _, src_pts, dst_pts = feat_match.match(img_right, img_left)
    
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(src_pts, dst_pts, F_mat, (img_right.shape[1], img_right.shape[0]), threshold = 3.0)
    
    return H1, H2, (h,w)