import numpy as np
import cv2
import glob
import files

# Termination criteria
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

imgpointsR, imgpointsL, objpoints, mtxR, distR, mtxL, distL, ChessImaR = files.read_calibration()
print("Calibration files successfully loaded")

# Call the two cameras
input("Connect the right camera and press any key ...")
devices = glob.glob("/dev/video?")
CamR = cv2.VideoCapture(devices[0])

input("Connect the left camera and press any key ...")
devices = glob.glob("/dev/video?")
CamL = cv2.VideoCapture(devices[0])

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

# is MLS = mtxL?

retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                           imgpointsL,
                                                           imgpointsR,
                                                           mtxL,
                                                           distL,
                                                           mtxR,
                                                           distR,
                                                           ChessImaR.shape[::-1],
                                                           criteria_stereo,
                                                           flags)

# StereoRectify function

# if 0 image croped, if 1 image nor croped
rectify_scale = 0
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, ChessImaR.shape[::-1], R, T, rectify_scale, (0, 0))

# initUndistortRectifyMap function
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, ChessImaR.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImaR.shape[::-1], cv2.CV_16SC2)

# Create StereoSGBM and prepare all parameters
window_size = 5
min_disp = 2
num_disp = 114 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32,
                               disp12MaxDiff=5,
                               preFilterCap=5,
                               P1=8 * 1 * window_size ** 2,
                               P2=32 * 1 * window_size ** 2)

# Used for the filtered image
stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000  # 80000
sigma = 1.8  # 1.8

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

while True:

    # Start Reading Camera images
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    # Rectify the images on rotation and alignement
    Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Convert from color (BGR) to gray
    grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

    # Compute the 2 images for the Depth_image
    dispL = stereo.compute(grayL, grayR)
    dispR = stereoR.compute(grayR, grayL)

    # Using the WLS filter

    # Disparity map left, left view, filtered_disparity map, disparity map right, ROI=rect (to be done)
    filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    # Change the Color of the Picture into an Ocean Color_Map
    filt_Color = cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN)

    cv2.imshow('Filtered Color Depth', cv2.resize(filteredImg, (320, 240)))
    cv2.imshow('Filtered L', cv2.resize(Left_nice, (320, 240)))
    cv2.imshow('Filtered R', cv2.resize(Right_nice, (320, 240)))

    # End the Programme
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
