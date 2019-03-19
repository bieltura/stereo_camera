import numpy as np
import cv2
import glob
import files

stereo_calibration = False

# Termination criteria
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

if stereo_calibration:
    imgpointsR, imgpointsL, objpoints, mtxR, distR, mtxL, distL, ChessImaR = files.read_stereo_calibration()
else:
    imgpointsR, objpoints, mtxR, distR, ChessImaR = files.read_single_calibration()
    imgpointsL = imgpointsR
    mtxL = mtxR
    distL = distR

print("Calibration files successfully loaded")
ChessImaR.shape[::-1]

newcameramtxR, roi = cv2.getOptimalNewCameraMatrix(mtxR, distR, ChessImaR.shape[::-1], 0, ChessImaR.shape[::-1])
newcameramtxL, roi = cv2.getOptimalNewCameraMatrix(mtxL, distL, ChessImaR.shape[::-1], 0, ChessImaR.shape[::-1])

# Capture the frame
frameR = cv2.imread('images/chessboard-R' + str(30) + '.png')
frameL = cv2.imread('images/chessboard-L' + str(30) + '.png')


dstR = cv2.undistort(frameR, mtxR, distR, None, newcameramtxR)
dstL = cv2.undistort(frameL, mtxL, distL, None, newcameramtxL)

cv2.imwrite('chessboard-R-cal-' + str(30) + '.png', dstR)
cv2.imwrite('chessboard-L-cal-' + str(30) + '.png', dstL)
