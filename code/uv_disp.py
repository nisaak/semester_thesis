import cv2
import numpy as np
import math

#define parameters for all functions
#shape of all images

#height, width = (720, 1280)
#height, width  = (360, 640) 
#height, width  = (768, 1024)
MAX_DISPARITY = 255


#def dir_filter():
    


def load_and_resize(disp_map, source_image):
    
    width, height = disp_map.shape


    src_img = np.copy(source_image)
    disp = np.copy(disp_map)
    
    src_img = cv2.resize(src_img, (width, height), interpolation =cv2.INTER_CUBIC)
    disp = cv2.resize(disp, (width, height), interpolation =cv2.INTER_CUBIC)

    return disp, src_img

def u_disp(disparity):
    
    height, width = disparity.shape
    
    #specify image shape for later use
    
    disp_map = np.copy(disparity).astype(np.uint8)
    
    
    
    #create U disparity histogram
    
    U_disp = np.zeros((MAX_DISPARITY, width),np.float)
    for u in range(width):
        U_disp[..., u] = cv2.calcHist(images = [disp_map[..., u]],channels = [0],mask = None, histSize = [MAX_DISPARITY], 
                                      ranges = [0, MAX_DISPARITY]).flatten() / np.float(width)
    
    uhist_vis = np.array(U_disp*255, np.uint8) #scale
    ublack_mask = uhist_vis < 5    #define threshold for black mask
    uhist_vis[ublack_mask] = 0  #set black mask to zero

    U_disp = uhist_vis
    
#    U_disp = cv2.normalize(U_disp, U_disp, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return U_disp
            
def v_disp(disparity):
    
    height, width = disparity.shape
    
    


    disp_map = np.copy(disparity)

    

    
    #    def v_disparity(self, image):
    V_disp = np.zeros((height, MAX_DISPARITY),np.float)
    for v in range(height):
        V_disp[v, ...] = cv2.calcHist(images = [disp_map[v, ...]],channels = [0],mask = None, histSize = [MAX_DISPARITY], 
                                      ranges = [0, MAX_DISPARITY]).flatten() / np.float(height)

    vhist_vis  = np.array(V_disp*255, np.uint8)
    vblack_mask = vhist_vis < 10
    vhist_vis[vblack_mask] = 0
    V_disp = vhist_vis
    
#    V_disp = cv2.normalize(V_disp, V_disp, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return V_disp

def u_hough(U_disp, method):
    
    
    
    #convert V_disp to right format
    cdst = np.zeros_like(U_disp)   
    threshold  = 150
    
    #specify which method to use, probabilistic or standard hough

    if method == 0:
        
        u_lines = cv2.HoughLines(U_disp, 2, np.pi/180, threshold)
        
        a,b,c = u_lines.shape
        for i in range(a):
            #if v_lines[i][0][0] > (-np.pi/2 + 10*np.pi/180) and v_lines[i][0][0] < (-10*np.pi/180):
            rho = u_lines[i][0][0]
            theta = u_lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0, y0 = a*rho, b*rho
            pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
            pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
            cv2.line(cdst,pt1,pt2,(255,255,255),1, cv2.LINE_AA)
    
    
    if method == 1:
        u_lines = cv2.HoughLinesP(U_disp, 2, np.pi/180, threshold, None, 30, 15) 
        
        #lines can extend outside rectangle making gradient false
        try:
            if u_lines.any:
                for i in range(0, len(u_lines)):
                    l = u_lines[i][0]
                    cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (255,255,0), 1, cv2.LINE_8)
            else:
                print('no lines detected')
        except AttributeError:
            print('no lines detected')

#    cdst = cv2.cvtColor(cdst, cv2.COLOR_BGR2GRAY)
    
    return cdst

    
def v_hough(V_disp, method):
    
    
    
    V_disp[0:int(V_disp.shape[0]/3),:] = 0 #make upper third black for more relevant line detection
    #convert V_disp to right format
    cdst = np.zeros_like(V_disp)   
    cdst_vert = np.zeros_like(V_disp)
    threshold  = 30
    #specify which method to use, probabilistic or standard hough

    if method == 0:
        
        v_lines = cv2.HoughLines(V_disp, 1, np.pi/180, threshold)
        
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
            cv2.line(cdst,pt1,pt2,(255,255,255),1, cv2.LINE_AA)
    
    
    if method == 1:
        v_lines = cv2.HoughLinesP(V_disp, 1, np.pi/180, threshold, None, 50, 50) #maybe only use longest line

        try:
            if v_lines.any:
                max_dist= -1.0
                max_l = np.zeros((4,1))
                for i in range(0, len(v_lines)): #iterate through all lines, find longest and print it
                    l = v_lines[i][0]
                    if np.abs(l[2]-l[0]) >= 10:
                        theta1 = (l[3]-l[1])
                        theta2 = (l[2]-l[0])
                        hyp = math.hypot(theta1, theta2)
                        if max_dist < hyp:
                            max_l = l
                            max_dist = hyp
                cv2.line(cdst, (max_l[0], max_l[1]), (max_l[2], max_l[3]), (255,255,0), 1, cv2.LINE_8)

            else:
                print('no lines detected')
        except AttributeError:
            print('no lines detected')
            
    cv2.imshow('cdst', cdst)

#    cdst = cv2.cvtColor(cdst, cv2.COLOR_BGR2GRAY)
    
    return cdst
    
def mask(disparity, cdst):
    
    
    height, width = disparity.shape
    
    cdst = cv2.resize(cdst, dsize = (255, height))
     
    threshold = 12
    
    disp_map = np.copy(disparity)
    
    #create mask for image
    
    v_line = np.zeros((height,1))
    for n in range(np.uint8(height/3),height):
        v_line[n] = np.argmax(cdst[n,:]) #this might not be the best way to do the disparity comparision
#        print(v_line[n])
#        v_line[n,0] = np.argwhere(cdst[n,:]).max(1)
    #calculate mask for v disparity

    mask_v_disp = np.zeros((height,width),np.uint8)
    for m in range(height):
        threshold -=10/(2*height)
        for n in range(width):


            if v_line[m] == 0:
                mask_v_disp[m,n] = 0

            elif (disp_map[m,n]  > (v_line[m]-threshold) and disp_map[m,n] < (v_line[m]+threshold)):
                mask_v_disp[m,n] = 255
            else:
                mask_v_disp[m,n] = 0

   
        
    return mask_v_disp
    
def refine_mask(mask):
    new_mask = np.zeros_like(mask)
    for val in np.unique(mask)[1:]:
        mask_for_labels = np.uint8(mask == val)
        labels, stats = cv2.connectedComponentsWithStats(mask_for_labels, 4)[1:3]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        new_mask[labels == largest_label] = val
        
    kernel = np.ones((3,3), np.uint8)
    kernel_close = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel)
#    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)
    new_mask = np.copy(closing)
    
    return new_mask
    

    
    
###IGNORE FOR NOW####
