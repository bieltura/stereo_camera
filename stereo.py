import numpy as np
import cv2
import files

# Termination criteria
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

imgpointsR, imgpointsL, objpoints, mtxR, distR, mtxL, distL, ChessImaR = files.read_calibration()

calibration_size = (640, 360)

print("Calibration files successfully loaded")

# Call the two cameras
CamR = cv2.VideoCapture(0)
CamL = cv2.VideoCapture(1)

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objectPoints=objpoints,
                                                           imagePoints1=imgpointsL,
                                                           imagePoints2=imgpointsR,
                                                           cameraMatrix1=mtxL,
                                                           distCoeffs1=distL,
                                                           cameraMatrix2=mtxR,
                                                           distCoeffs2=distR,
                                                           imageSize=ChessImaR.shape[::-1],
                                                           criteria=criteria_stereo,
                                                           flags=flags)

# StereoRectify function: returns rotation matrix and projection matrix

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

    # Resize to the calibration
    frameR = cv2.resize(frameR, calibration_size)
    frameL = cv2.resize(frameL, calibration_size)

    # Rectify the images on rotation and alignement
    Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Convert from color (BGR) to gray
    grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

    # Filter the noise to make the stereo match in low light conditions
    grayR = cv2.fastNlMeansDenoising(grayR, None, h=4)
    grayL = cv2.fastNlMeansDenoising(grayL, None, h=4)

    # Compute the 2 images for the Depth_image
    dispL = stereo.compute(grayL, grayR)
    dispR = stereoR.compute(grayR, grayL)

    # Disparity map left, left view, filtered_disparity map, disparity map right, ROI=rect (to be done)
    filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    # Change the Color of the Picture into an Ocean Color_Map
    filteredImg = cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN)

    cv2.imshow('Filtered Color Depth', cv2.resize(filteredImg, calibration_size))
    cv2.imshow('Both rectified and noise filtered', np.hstack([grayL, grayR]))
    cv2.imshow('Real capture', np.hstack([frameL, frameR]))

    # End the Programme
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
