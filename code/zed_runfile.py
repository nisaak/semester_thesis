import cv2
import pyzed.sl as sl
import sys
import os
sys.path.append(os.path.abspath("/home/nisaak/Documents/st/semester_thesis/code/"))
import uv_disp
import numpy as np
import math
import time

timestr = time.strftime("%Y%m%d-%H%M%S")




camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1



def main():
    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.RESOLUTION_VGA
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_MEDIUM  # Use PERFORMANCE depth mode
    init.coordinate_units = sl.UNIT.UNIT_METER  # Use milliliter units (for depth measurements)
    init.depth_minimum_distance = 0.3
    cam = sl.Camera()
    cam.set_depth_max_range_value(40)
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD  # Use STANDARD sensing mode

    mat = sl.Mat()
    disparity = sl.Mat()
    depth = sl.Mat()
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    #specify all video to be written
    mask_video=cv2.VideoWriter('./videos/mask_'+ timestr +'.avi',fourcc, 10,(672, 376),0)
    disparity_video=cv2.VideoWriter('./videos/disparity_'+ timestr +'.avi',fourcc, 10,(672, 376),0)
    v_disparity_video=cv2.VideoWriter('./videos/v_disp_'+ timestr +'.avi',fourcc, 10,(255, 376),0)




    print_camera_information(cam)
    print_help()

    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.VIEW_LEFT)
            cam.retrieve_measure(disparity, sl.MEASURE.MEASURE_DISPARITY) #retrieve disparity map
            cam.retrieve_image(depth, sl.VIEW.VIEW_DEPTH)
            cv2.imshow('depth',depth.get_data())
            
            
            
            ###MASK CALCULATIONS

            u = uv_disp.UV_disp()
           
            disp_np = np.copy(disparity.get_data()) #copy and convert to numpy array..! there are issues because normalization with nan and inf
#            disp_np = np.copy(depth.get_data())
#            disp_np = cv2.cvtColor(disp_np, cv2.COLOR_BGRA2GRAY)
#            info = np.finfo(disp_np.dtype)
#            disp_np[np.isnan(disp_np)] = 0 #this is need because else the images becomes black after normalization
            disp_np[np.isinf(disp_np)] = 0
            cv2.normalize(disp_np, disp_np, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F) #normalizing makes flickering artifacts, change norm type
            disp_np = disp_np.astype(np.uint8)



            """
            TODO:
            Currently, disparity map is normalized according to uint8 range, try to find a way to preserve original disparity map values and compare with them
            """
            V_disp = u.v_disp(disp_np) #compute v-disp histogram
#            U_disp = u.u_disp(disp_np)
            
            
            #maybe remove noise in disparity map
            
            v_houghlines = u.v_hough(V_disp)
#            u.backprop(V_disp, disp_np, v_houghlines)

#            u_houghlines = u.u_hough(U_disp)
            disp_np = cv2.resize(disp_np, dsize =(0,0),fx = 0.5, fy= 0.5, interpolation = cv2.INTER_AREA)
            
#            obst_mask = u.mask_obstacles(disp_np, u_houghlines)
            
            floor_mask = u.mask(disp_np, v_houghlines)
            floor_mask = cv2.resize(floor_mask, dsize =(0,0),fx = 2, fy= 2,interpolation = cv2.INTER_CUBIC)
            
            refined_mask = u.refine_mask(floor_mask) #astype uint8
#            refined_mask = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)

            cam_img = mat.get_data() #astype uint8 4 channels RGBA
            cam_img = cv2.cvtColor(cam_img, cv2.COLOR_RGBA2GRAY)
#            cam_img = cv2.cvtColor(cam_img, cv2.COLOR_RGBA2BGR)
            mask_on_img = cv2.addWeighted(cam_img,1,refined_mask,0.5, 5)
            disp_np = cv2.resize(disp_np, dsize= (0,0), fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
#            obst_mask = cv2.resize(obst_mask, dsize= (0,0), fx=4, fy=4)
#            obst_on_img = cv2.addWeighted(cam_img,1,obst_mask,0.5, 5)
            
            mask_video.write(mask_on_img)
            disparity_video.write(disp_np)
            v_disparity_video.write(V_disp)
            
            
            cv2.imshow('disp',disp_np)
#            cv2.imshow('u-hough',u_houghlines)
#            cv2.imshow('obst_mask', obst_mask)
            cv2.imshow('maskonimg', mask_on_img)
#            cv2.imshow('obstonimg', obst_on_img)
            cv2.imshow('V-disparity',V_disp)
#            cv2.imshow('U-disparity',U_disp)
#            cv2.imshow("ZED", mat.get_data())
#            cv2.imshow('refined_mask', refined_mask)

            
            key = cv2.waitKey(5)
            settings(key, cam, runtime, mat)
        else:
            key = cv2.waitKey(5)
    
    
    #release all videos
    mask_video.release()
    disparity_video.release()
    v_disparity_video.release()
    
    
    cv2.destroyAllWindows()

    cam.close()
    print("\nFINISH")


def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(round(cam.get_resolution().width, 2), cam.get_resolution().height))
    print("Camera FPS: {0}.".format(cam.get_camera_fps()))
    print("Firmware: {0}.".format(cam.get_camera_information().firmware_version))
    print("Serial number: {0}.\n".format(cam.get_camera_information().serial_number))


def print_help():
    print("Help for camera setting controls")
    print("  Increase camera settings value:     +")
    print("  Decrease camera settings value:     -")
    print("  Switch camera settings:             s")
    print("  Reset all parameters:               r")
    print("  Record a video:                     z")
    print("  Quit:                               q\n")


def settings(key, cam, runtime, mat):
    if key == 43:  # for '+' key
        current_value = cam.get_camera_settings(camera_settings)
        cam.set_camera_settings(camera_settings, current_value + step_camera_settings)
        print(str_camera_settings + ": " + str(current_value + step_camera_settings))
    elif key == 45:  # for '-' key
        current_value = cam.get_camera_settings(camera_settings)
        if current_value >= 1:
            cam.set_camera_settings(camera_settings, current_value - step_camera_settings)
            print(str_camera_settings + ": " + str(current_value - step_camera_settings))

    elif key == 122:  # for 'z' key
        record(cam, runtime, mat)



def record(cam, runtime, mat):
    vid = sl.ERROR_CODE.ERROR_CODE_FAILURE
    out = False
    while vid != sl.ERROR_CODE.SUCCESS and not out:
        filepath = input("Enter filepath name: ")
        vid = cam.enable_recording(filepath)
        print(repr(vid))
        if vid == sl.ERROR_CODE.SUCCESS:
            print("Recording started...")
            out = True
            print("Hit spacebar to stop recording: ")
            key = False
            while key != 32:  # for spacebar
                err = cam.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(mat)
                    cv2.imshow("ZED", mat.get_data())
                    key = cv2.waitKey(5)
                    cam.record()
        else:
            print("Help: you must enter the filepath + filename + SVO extension.")
            print("Recording not started.")
    cam.disable_recording()
    print("Recording finished.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()