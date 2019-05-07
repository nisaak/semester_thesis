import cv2
import pyzed.sl as sl
import sys
import os
sys.path.append(os.path.abspath("/home/nisaak/Documents/st/semester_thesis/code/"))
import uv_disp
import numpy as np
import math

camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1



def main():
    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.RESOLUTION_VGA
    init.camera_fps = 60
    init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
    init.coordinate_units = sl.UNIT.UNIT_MILLIMETER  # Use milliliter units (for depth measurements)
    init.depth_minimum_distance = 300
    cam = sl.Camera()
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
            
            disp_np = np.copy(disparity.get_data()) #copy and convert to numpy array..! there are issues because normalization with nan and inf
            info = np.finfo(disp_np.dtype)
            disp_np[np.isnan(disp_np)] = 0 #this is need because else the images becomes black after normalization
            disp_np[np.isinf(disp_np)] = 0

            cv2.normalize(disp_np, disp_np, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F) #normalizing makes flickering artifacts
            disp_np = disp_np.astype(np.uint8)
            V_disp = uv_disp.v_disp(disp_np) #compute v-disp histogram
            
            houghlines = uv_disp.v_hough(V_disp, 1)
            disp_np = cv2.resize(disp_np, dsize =(0,0),fx = 0.5, fy= 0.5, interpolation = cv2.INTER_AREA)

            
            floor_mask = uv_disp.mask(disp_np, houghlines)
            floor_mask = cv2.resize(floor_mask, dsize =(0,0),fx = 2, fy= 2,interpolation = cv2.INTER_AREA)
            
            refined_mask = uv_disp.refine_mask(floor_mask) #astype uint8

            cam_img = mat.get_data() #astype uint8 4 channels RGBA
            cam_img = cv2.cvtColor(cam_img, cv2.COLOR_RGBA2GRAY)
            
            
            mask_on_img = cv2.addWeighted(cam_img,1,refined_mask,0.5, 5)


            
            
            disp_np = cv2.resize(disp_np, dsize= (0,0), fx=2, fy=2)
            cv2.imshow('maskonimg', mask_on_img)
            cv2.imshow('disp_for_display', disp_np)
            cv2.imshow('V-disparity',V_disp)
#            cv2.imshow("ZED", mat.get_data())
#            cv2.imshow('refined_mask', refined_mask)

            
            key = cv2.waitKey(5)
            settings(key, cam, runtime, mat)
        else:
            key = cv2.waitKey(5)
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