import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math


def uv_disp(disp, img_right):
        
    """
    path = "/home/nisaak/Documents/semester_thesis/code_test/test_pictures/"
    #imgpath = path  + "Cars_gnd.png"
    imgpath = path  + "disparity.png"
    #imgpath = path  + "disp.jpg"
    #imgpath = path + "test_2.png"
    
    
    path_left = '/home/nisaak/Documents/semester_thesis/export_test/day_cloudy1/left/frame0161.jpg'
    source_img = np.uint8(cv2.imread(path_left, cv2.IMREAD_GRAYSCALE))
    source_img = cv2.resize(source_img, (420,289), interpolation =cv2.INTER_CUBIC)
    """
    source_img = disp
    source_img = cv2.resize(source_img, (420,289), interpolation =cv2.INTER_CUBIC)
    print(source_img.shape)
    
    
    #
    #img = np.asarray(cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE))
    img = img_right
    img = cv2.resize(img, (420,289), interpolation =cv2.INTER_CUBIC)
    #cv2.imshow('img', img)
    #    cv2.waitKey()
    (img_height, img_width) = np.shape(img)
    print(img_height,img_width)
    
    
    MAX_DISPARITY = 200 
    MIN_DISPARITY = 0
    #    def u_disparity(self):
    U_disp = np.zeros((MAX_DISPARITY, img_width),np.float)
    for u in range(img_width):
        U_disp[..., u] = cv2.calcHist(images = [img[..., u]],channels = [0],mask = None, histSize = [MAX_DISPARITY], 
                                      ranges = [0, MAX_DISPARITY]).flatten() / np.float(img_width)
    uhist_vis = np.array(U_disp*255, np.uint8)
    ublack_mask = uhist_vis < 30
    uhist_vis[ublack_mask] = 0
    U_disp = uhist_vis
    #MAX_U = np.max(U_disp)
    #U_disp = np.multiply(U_disp, 1./MAX_U)
    
    #ret,U_disp = cv2.threshold(U_disp,0.1,1,cv2.THRESH_BINARY)
    #U_disp = cv2.applyColorMap(U_disp,cv2.COLORMAP_JET)
    cv2.imshow('U_disp', U_disp)
    
    #ret, labels = cv2.connectedComponents(img = U_disp, connectivity = 8)
    #cv2.imshow('labels',labels)
    
    
    
    #    MAX2 = np.max(U_disp)
    
        
    #    def v_disparity(self, image):
    V_disp = np.zeros((img_height, MAX_DISPARITY),np.float)
    for v in range(img_height):
        V_disp[v, ...] = cv2.calcHist(images = [img[v, ...]],channels = [0],mask = None, histSize = [MAX_DISPARITY], 
                                      ranges = [0, MAX_DISPARITY]).flatten() / np.float(img_height)
    
    #MAX_V = np.max(V_disp)
    #V_disp = V_disp/MAX_V
    #ret,V_disp = cv2.threshold(V_disp,0.25,1,cv2.THRESH_BINARY)
    vhist_vis  = np.array(V_disp*255, np.uint8)
    vblack_mask = vhist_vis < 20
    vhist_vis[vblack_mask] = 0
    V_disp = vhist_vis
     
    cv2.imshow('V_disp', V_disp)
    #cv2.waitKey()
    
    V_disp = V_disp.astype(np.uint8)
    U_disp = U_disp.astype(np.uint8)
    
    #convert V_disp to right format
    cdst = cv2.cvtColor(V_disp, cv2.COLOR_GRAY2BGR)
    
    #convert U_disp to right format
    cdst_u = cv2.cvtColor(U_disp, cv2.COLOR_GRAY2BGR)
    
    #linedetection v disp
    v_lines = cv2.HoughLinesP(V_disp, 1, np.pi/180, 100, None, 100, 3) 
    #TODO make houghdetection more robust
    
    
    #lines can extend outside rectangle making gradient false
    threshold_line = 100
    if v_lines: 
        for i in range(0, len(v_lines)):
            l = v_lines[i][0]
        #    if l[3]-l[1] > threshold_line:
        #    angle = math.atan(math.fabs((l[2]-l[1])/(l[3]-l[0])))
        #    if (angle > 280*np.pi/180 and angle < -10*np.pi/180):
            cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (255,255,0), 1, cv2.LINE_AA)
    else:
        print("no lines detected")
    
    
    """
    a,b,c = v_lines.shape
    for i in range(a):
    #    if v_lines[i][0][0] > (-np.pi/2 + 10*np.pi/180) and v_lines[i][0][0] < (-10*np.pi/180):
        rho = v_lines[i][0][0]
        theta = v_lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = a*rho, b*rho
        pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
        pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
        cv2.line(cdst,pt1,pt2,(255,255,255),2, cv2.LINE_AA)
    """
    
    cdst = cv2.cvtColor(cdst, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('cdst', cdst)
    
    
    
    """
    #linedetection udisp
    u_lines = cv2.HoughLinesP(U_disp, 1, np.pi/180, 10,None,10,20) 
    
    for i in range(0, len(u_lines)):
        l = u_lines[i][0]
        cv2.line(cdst_u, (l[0], l[1]), (l[2], l[3]), (255,255,0), 1, cv2.LINE_AA)
    
       
    cdst_u = cv2.cvtColor(cdst_u, cv2.COLOR_BGR2GRAY)
    cv2.imshow('cdst_u',cdst_u)
    """
    #create mask for image
    v_line = np.zeros((img_height,1))
    for n in range(np.uint8(img_height/2),img_height):
        v_line[n] = np.argmax(cdst[n,:])
    
    #    v_lines[n,0] = np.argwhere(cdst[n,:]).max(1)
    #print(line)
    threshold = 7
    
    lower =0 
    #np.uint8(img_height/2)
    #calculate mask for v disparity
    mask_v_disp = np.zeros((img_height,img_width),np.uint8)
    for m in range(img_height):
        for n in range(img_width):
            if (img[m,n] > (v_line[m]-threshold) and img[m,n] < (v_line[m]+threshold)):
                mask_v_disp[m,n] = 255
    #            img[m,n]=0
            else:
                mask_v_disp[m,n] = 0
    imgandmask = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
    #mask_v_disp = cv2.cvtColor(mask_v_disp, cv2.COLOR_GRAY2BGR)
    print(mask_v_disp.shape)
    imgandmask = cv2.addWeighted(source_img,0.5,mask_v_disp,0.5,1)
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
    
    #matplotlib.image.imsave('img',imgandmask)
    #matplotlib.image.imsave('v_disp',V_disp)
    #matplotlib.image.imsave('v_line',cdst)
    #matplotlib.image.imsave('u_disp',U_disp)
    #matplotlib.image.imsave('u_line',cdst_u)
    cv2.imshow('mask',imgandmask)
    #cv2.imshow('mask_u',mask_u_disp)
    
    cv2.imshow('img',img)
    
    cv2.waitKey()
    #print(type(V_disp))
    #print(type(v_lines))
    #cv2.imwrite(cdst, u_lines)