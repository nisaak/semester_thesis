import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

#define parameters for all functions
#shape of all images

height, width = (720, 1280)
#height, width  = (360, 640) 
#height, width  = (768, 1024)
MAX_DISPARITY = 160


def load_and_resize(disp_map, source_image):

    src_img = np.copy(source_image)
    disp = np.copy(disp_map)
    
    src_img = cv2.resize(src_img, (width, height), interpolation =cv2.INTER_CUBIC)
    disp = cv2.resize(disp, (width, height), interpolation =cv2.INTER_CUBIC)

    return disp, src_img

def u_disp(disp_map):
    
    #specify image shape for later use
    
    
    #create U disparity histogram
    
    U_disp = np.zeros((MAX_DISPARITY, width),np.float)
    for u in range(width):
        U_disp[..., u] = cv2.calcHist(images = [disp_map[..., u]],channels = [0],mask = None, histSize = [MAX_DISPARITY], 
                                      ranges = [0, MAX_DISPARITY]).flatten() / np.float(width)
    
    uhist_vis = np.array(U_disp*255, np.uint8) #scale
    ublack_mask = uhist_vis < 30    #define threshold for black mask
    uhist_vis[ublack_mask] = 0  #set black mask to zero
    U_disp = uhist_vis
    
    U_disp = U_disp.astype(np.uint8)
    
    return U_disp
    
def v_disp(disparity):
    
    disp_map = np.copy(disparity).astype(np.float32)
    
    #    def v_disparity(self, image):
    V_disp = np.zeros((height, MAX_DISPARITY),np.float)
    for v in range(height):
        V_disp[v, ...] = cv2.calcHist(images = [disp_map[v, ...]],channels = [0],mask = None, histSize = [MAX_DISPARITY], 
                                      ranges = [0, MAX_DISPARITY]).flatten() / np.float(height)
        
    vhist_vis  = np.array(V_disp*255, np.uint8)
    vblack_mask = vhist_vis < 20
    vhist_vis[vblack_mask] = 0
    V_disp = vhist_vis
    
    V_disp = V_disp.astype(np.uint8)
    
    return V_disp
    
def v_hough(V_disp, method):
    
    
    
    V_disp[0:int(height/2),:] = 0 #make upper half black for more relevant line detection
    cv2.imshow('blackedvdisp',V_disp)
    #convert V_disp to right format
    cdst = cv2.cvtColor(V_disp, cv2.COLOR_GRAY2BGR)
    
    threshold  = 100
    
    #specify which method to use, probabilistic or standard hough

    if method == 0:
        
        v_lines = cv2.HoughLines(V_disp, 2, np.pi/180, threshold)
        
        a,b,c = v_lines.shape
        for i in range(a):
            #if v_lines[i][0][0] > (-np.pi/2 + 10*np.pi/180) and v_lines[i][0][0] < (-10*np.pi/180):
            rho = v_lines[i][0][0]
            theta = v_lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0, y0 = a*rho, b*rho
            pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
            pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
            cv2.line(cdst,pt1,pt2,(255,255,255),2, cv2.LINE_AA)
    
    
    if method == 1:
        v_lines = cv2.HoughLinesP(V_disp, 2, np.pi/180, threshold, None, 100, 3) 
        
        #lines can extend outside rectangle making gradient false
        try:
            if v_lines.any:
                for i in range(0, len(v_lines)):
                    l = v_lines[i][0]
                    cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (255,255,0), 1, cv2.LINE_AA)
            else:
                print('no lines detected')
        except AttributeError:
            print('no lines detected')

    cdst = cv2.cvtColor(cdst, cv2.COLOR_BGR2GRAY)
    
    return cdst
    
def mask(disp_map, cdst):
        
    threshold = 7

    
    #create mask for image
    
    v_line = np.zeros((height,1))
    for n in range(np.uint8(height/2),height):
        v_line[n] = np.argmax(cdst[n,:])
    
    #v_lines[n,0] = np.argwhere(cdst[n,:]).max(1)
    
    #calculate mask for v disparity

    mask_v_disp = np.zeros((height,width),np.uint8)
    for m in range(height):
        for n in range(width):
            if (disp_map[m,n] > (v_line[m]-threshold) and disp_map[m,n] < (v_line[m]+threshold)):
                mask_v_disp[m,n] = 255
            else:
                mask_v_disp[m,n] = 0
   
        
    return mask_v_disp
    
    
    

    
    
###IGNORE FOR NOW####
    
    
"""
#convert U_disp to right format

cdst_u = cv2.cvtColor(U_disp, cv2.COLOR_GRAY2BGR)

#linedetection udisp
u_lines = cv2.HoughLinesP(U_disp, 1, np.pi/180, 10,None,10,20) 

for i in range(0, len(u_lines)):
    l = u_lines[i][0]
    cv2.line(cdst_u, (l[0], l[1]), (l[2], l[3]), (255,255,0), 1, cv2.LINE_AA)

   
cdst_u = cv2.cvtColor(cdst_u, cv2.COLOR_BGR2GRAY)
cv2.imshow('cdst_u',cdst_u)
"""

"""

mask_u_disp = np.zeros((img_height,img_width))

u_line = np.zeros((1,img_width),np.uint8)
try:
    for n in range((img_width)):
        u_line[0,n] = np.max(np.nonzero(cdst_u[:,n]))
except ValueError:
    pass
#    u_line[0,n] = np.argwhere(cdst_u[:,n]).max(0)
threshold_obj = 5





#calculate mask for u disparity add disparity colour
mask_u_disp = np.zeros((img_height,img_width))

for n in range(img_width):
    for m in range(img_height):
        if img[m,n] > 5:
            if img[m,n] < (u_line[0,m]-threshold_obj) and img[m,n] > (u_line[0,m]+threshold_obj):
                mask_u_disp[m,n] = 255
            else:
                mask_u_disp[m,n] = 0
#            if (np.any(np.where(cdst_u[:,n] > 0))-threshold_obj) >= img[m,n] <= (np.any(np.where(cdst_u[:,n] > 0))+threshold_obj):
#                    mask_u_disp[m,n] = 1
#            else:
#                mask_u_disp[m,n] = 0

"""