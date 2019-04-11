import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math



path = "/home/nisaak/Documents/semester_thesis/code_test/test_pictures/"
#imgpath = path  + "Cars_gnd.png"
#imgpath = path  + "disp_map.png"
imgpath = path  + "test_2.png"
#imgpath = path + "disp.jpg"

#
img = np.asarray(cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE))
#img = cv2.resize(img,None,fx=2,fy=2, interpolation =cv2.INTER_CUBIC)
#    cv2.imshow('img', img)
#    cv2.waitKey()
(img_height, img_width) = np.shape(img)
np.uint8(img_height)
np.uint8(img_width)


MAX_DISPARITY = 200
#    def u_disparity(self):
U_disp = np.zeros((MAX_DISPARITY, img_width))
for u in range(img_width):
    U_disp[..., u] = cv2.calcHist(images = [img[..., u]],channels = [0],mask = None, histSize = [MAX_DISPARITY], 
                                  ranges = [0, MAX_DISPARITY]).flatten() / float(img_width)
MAX = np.max(U_disp)
U_disp = U_disp/MAX
#cv2.imshow('U_disp', U_disp)

#    MAX2 = np.max(U_disp)

    
#    def v_disparity(self, image):
V_disp = np.zeros((img_height, MAX_DISPARITY))
for v in range(img_height):
    V_disp[v, ...] = cv2.calcHist(images = [img[v, ...]],channels = [0],mask = None, histSize = [MAX_DISPARITY], 
                                  ranges = [0, MAX_DISPARITY]).flatten() / float(img_height)
    

#cv2.imshow('V_disp', V_disp)

V_disp = V_disp.astype(np.uint8)


U_disp = U_disp.astype(np.uint8)
cdst = cv2.cvtColor(V_disp, cv2.COLOR_GRAY2BGR)

v_lines = cv2.HoughLines(V_disp, 1, np.pi/180, 0)
#print(v_lines) 
#TODO make houghdetection more robust
for rho,theta in v_lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(cdst,(x1,y1),(x2,y2),(0,0,255),2)

cdst = cv2.cvtColor(cdst, cv2.COLOR_BGR2GRAY)
#cv2.imshow('wat', cdst)
    
#create mask for image
line = np.zeros((img_height,1))
for n in range(np.uint8(img_height/2),img_height):
    line[n] = np.argmax(cdst[n,:])
print(line)
threshold = 4
mask = np.zeros((img_height,img_width))
for m in range(img_height):
    for n in range(img_width):
        if (img[m,n] > (line[m]-threshold) and img[m,n] < (line[m]+threshold)):
            mask[m,n] = 1
            img[m,n] = 255
        else:
            mask[m,n] = 0

#matplotlib.image.imsave('test_mask',img)
cv2.imshow('mask',mask)
cv2.imshow('img',img)

cv2.waitKey()
#print(type(V_disp))
#print(type(v_lines))
#cv2.imwrite(cdst, u_lines)