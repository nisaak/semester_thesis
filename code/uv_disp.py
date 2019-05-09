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
    
class UV_disp:

    def load_and_resize(self,disp_map, source_image):
        
        width, height = disp_map.shape
    
    
        src_img = np.copy(source_image)
        disp = np.copy(disp_map)
        
        src_img = cv2.resize(src_img, (self.width, self.height), interpolation =cv2.INTER_CUBIC)
        disp = cv2.resize(disp, (self.width, self.height), interpolation =cv2.INTER_CUBIC)
    
        return disp, src_img
    
    def u_disp(self,disparity):
        
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
                
    def v_disp(self,disparity):
        
        height, width = disparity.shape
        
        
    
    
        disp_map = np.copy(disparity).astype(np.uint8)
    
        
    
        
        #    def v_disparity(self, image):
        V_disp = np.zeros((height, MAX_DISPARITY),np.float)
        for v in range(height):
            V_disp[v, ...] = cv2.calcHist(images = [disp_map[v, ...]],channels = [0],mask = None, histSize = [MAX_DISPARITY], 
                                          ranges = [0, MAX_DISPARITY]).flatten() / np.float(height)
    
        cv2.normalize(V_disp, V_disp, 0 ,255, cv2.NORM_MINMAX, cv2.CV_32F)
#        V_disp = V_disp.astype(np.uint8)
        vhist_vis  = np.array(V_disp*255, np.uint8)
#        vblack_mask = vhist_vis < 5
#        vhist_vis[vblack_mask] = 0
        V_disp = vhist_vis
        
    #    V_disp = cv2.normalize(V_disp, V_disp, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return V_disp
    
    def u_hough(self, U_disparity):
        
        
        U_disp = np.copy(U_disparity)
        U_disp[U_disp.shape[0]-5:U_disp.shape[0],:] = 0
        U_disp[0:5,:] = 0

        
        #convert V_disp to right format
        cdst_hor = np.zeros_like(U_disp)
        max_linelen = 150
        threshold_hor = 120
        v_lines_hor = cv2.HoughLinesP(U_disp, 1, np.pi/180, threshold_hor, None, 30, 5) #maybe only use longest line
        try:
            if v_lines_hor.any:
                for i in range(0, len(v_lines_hor)): #iterate through all lines, find longest and print it
                    l = v_lines_hor[i][0]
                    if np.abs(l[2]-l[0]) < max_linelen and np.abs(l[3]-l[1]) < 5:
                      cv2.line(cdst_hor, (l[0], l[1]), (l[2], l[3]), (255,255,0), 1, cv2.LINE_8)
            else:
              print('no lines detected')
        except AttributeError:
            print('no lines detected')
                
        return cdst_hor
    
        
    def v_hough(self,V_disparity):
        
        V_disp = np.copy(V_disparity)
        
        V_disp[0:int(V_disp.shape[0]/3),:] = 0 #make upper third black for more relevant line detection
        #convert V_disp to right format
        cdst = np.zeros_like(V_disp)   
        threshold_long  = 100
        v_lines = cv2.HoughLinesP(V_disp, 1, np.pi/180, threshold_long, None, 50, 5) #maybe only use longest line
        self.gradient = 0 #initialize gradient
        try:
            if v_lines.any:
                max_dist= -1.0
                max_l = np.zeros((4,1))
                for i in range(0, len(v_lines)): #iterate through all lines, find longest and print it
                    l = v_lines[i][0]
                    if np.abs(l[2]-l[0]) >= 5:
                        theta1 = (l[3]-l[1])
                        theta2 = (l[2]-l[0])
                        hyp = math.hypot(theta1, theta2)
                        if max_dist < hyp:
                            max_l = l
                            max_dist = hyp
                            self.gradient = (max_l[3]-max_l[1])/(max_l[0]-max_l[2])
                cv2.line(cdst, (max_l[0], max_l[1]), (max_l[2], max_l[3]), (255,255,0), 1, cv2.LINE_8)

            else:
                print('no lines detected')
        except AttributeError:
            print('no lines detected')
                
        cv2.imshow('cdst', cdst)
    
        return cdst
      
      
    def v_hough_peaks(self, V_disparity):
        V_disp = np.copy(V_disparity)
        
#        V_disp[0:int(V_disp.shape[0]/3),:] = 0 #make upper third black for more relevant line detection
        #convert V_disp to right format
        cdst_vert = np.zeros_like(V_disp)
        max_linelen = 100
        threshold_vert = 60
        v_lines_vert = cv2.HoughLinesP(V_disp, 1, np.pi/180, threshold_vert, None, 50, 10) #maybe only use longest line
        try:
            if v_lines_vert.any:
                for i in range(0, len(v_lines_vert)): #iterate through all lines, find longest and print it
                    l = v_lines_vert[i][0]
                    if np.abs((l[3]-l[1])) < max_linelen and np.abs(l[2]-l[0]) < 5:
                      cv2.line(cdst_vert, (l[0], l[1]), (l[2], l[3]), (255,255,0), 1, cv2.LINE_8)
            else:
              print('no lines detected')
        except AttributeError:
            print('no lines detected')
                
        return cdst_vert
      
      
    def mask(self,disparity, cdst):
        
        
        height, width = disparity.shape
            
        cdst = cv2.resize(cdst, dsize = (255, height))
         
        threshold_baseline = 15 #adjust threshold according to line gradient
        grad = self.gradient
#        print(self.gradient)
        k = math.pi/2 - math.atan(grad) #gradient dependent threshold
        threshold = math.cos(k) * threshold_baseline
        
        disp_map = np.copy(disparity)
        
        #create mask for image
        
        v_line = np.zeros((height,1))
        for n in range(np.uint8(height/3),height):
            v_line[n] = np.argmax(cdst[n,:]) #this might not be the best way to do the disparity comparision
    #        v_line[n,0] = np.argwhere(cdst[n,:]).max(1)
        #calculate mask for v disparity
    
        mask_v_disp = np.zeros((height,width),np.uint8)
        for m in range(height):
            for n in range(width):   
                if v_line[m] == 0:
                    mask_v_disp[m,n] = 0
    
                elif (disp_map[m,n]  >= (v_line[m]-threshold) and disp_map[m,n] <= (v_line[m]+threshold)):
                    mask_v_disp[m,n] = 255
                else:
                    mask_v_disp[m,n] = 0
    
       
            
        return mask_v_disp
      
    def mask_obstacles(self,disparity, cdst):
        
        
        height, width = disparity.shape
            
        cdst = cv2.resize(cdst, dsize = (width, 255))


         
        threshold = 5 #adjust threshold according to line gradient
        
        
        disp_map = np.copy(disparity)
        
        #create mask for image
        
        u_line = np.zeros((1,width))
        for n in range(width):
#            u_line[0,n] = UV_disp.last_nonzero(cdst[:,n],0)
            u_line[0,n] = (cdst[:,n]!=0).argmax(axis=0)
            
            
        mask_obst = np.zeros((height,width),np.uint8)
        for m in range(height):
            for n in range(width):
                if u_line[0,n] == 0:
                    mask_obst[m,n] = 0
                elif (disp_map[m,n]  >= (u_line[0,n]-threshold) and disp_map[m,n] <= (u_line[0,n]+threshold)):
                    mask_obst[m,n] = 255
                else:
                    mask_obst[m,n] = 0
    
       
            
        return mask_obst
        
    def refine_mask(self, mask):
        new_mask = np.zeros_like(mask)
        for val in np.unique(mask)[1:]:
            mask_for_labels = np.uint8(mask == val)
            labels, stats = cv2.connectedComponentsWithStats(mask_for_labels, 4)[1:3]
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            new_mask[labels == largest_label] = val
            
        kernel = np.ones((3,3), np.uint8)
        kernel_close = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel)
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)
        new_mask = np.copy(closing)
        
        return new_mask
        
    def similarity_compute(self, cdst_v, cdst_u):
        cv2.blur(cdst_v, (7,7))
        cv2.blur(cdst_u, (7,7))
        obst = np.matmul(cdst_v, cdst_u)
        return obst*255
      
      
    def last_nonzero(arr, axis, invalid_val=-1):
      mask = arr!=0
      val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
      return np.where(mask.any(axis=axis), val, invalid_val)
        
    ###IGNORE FOR NOW####
