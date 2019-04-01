import cv2
import numpy as np
import files

# Variables for calibration
num_pictures = 50
chessboard_dim = (9, 6)
calibration_size = (640, 360)

# Image variables
ChessImaR = None

# Prepare object points (index of chessboard points)
objp = np.zeros((chessboard_dim[0] * chessboard_dim[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_dim[0], 0:chessboard_dim[1]].T.reshape(-1, 2)

# Termination criteria
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all images
objpoints = []   # 3d points in real world space
imgpointsR = []   # 2d points in image plane
imgpointsL = []

print("Files for calibration generated")
print("Loading chessboard pictures")

for pic in range(num_pictures):

    # Capture the frame calibration_size
    frameR = cv2.imread('images/chessboard-R' + str(pic) + '.png')
    frameL = cv2.imread('images/chessboard-L' + str(pic) + '.png')

    # Resize it to the calibration size
    frameR = cv2.resize(frameR, calibration_size)
    frameL = cv2.resize(frameL, calibration_size)

    # Convert to gray scale
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_dim, None)
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_dim, None)

    # If found, add object points, image points (after refining them)
    if retR and retL:
        objpoints.append(objp)

        corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria_stereo)
        imgpointsR.append(cornersR)

        corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria_stereo)
        imgpointsL.append(cornersL)

        print('Loaded chessboard {0}/{1}'.format(pic, num_pictures))

# Calibration
_, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
_, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)

print("Right camera: A matrix: \n{}".format(mtxR))
print("Distoriton coefficients: \n{}".format(distR))

print("Left camera: A matrix: \n{}".format(mtxL))
print("Distoriton coefficients: \n{}".format(distL))

files.write_calibration(imgpointsL, imgpointsR, objpoints, mtxR, distR, mtxL, distL, grayR)

print("Calibration ended")
