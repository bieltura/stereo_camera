import cv2
import files
import numpy as np

imgpointsR, imgpointsL, objpoints, mtxR, distR, mtxL, distL, ChessImaR = files.read_calibration()

calibration_size = (640, 360)

# Termination criteria
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print("Calibration files successfully loaded")

# Call the two cameras
CamR = cv2.VideoCapture(0)
newcameramtxR, roi = cv2.getOptimalNewCameraMatrix(mtxR, distR, ChessImaR.shape[::-1], 0, ChessImaR.shape[::-1])

CamL = cv2.VideoCapture(1)
newcameramtxL, roi = cv2.getOptimalNewCameraMatrix(mtxL, distL, ChessImaR.shape[::-1], 0, ChessImaR.shape[::-1])

while True:

    key = cv2.waitKey(1)

    # Capture the frame
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    frameR = cv2.resize(frameR, calibration_size)
    frameL = cv2.resize(frameL, calibration_size)

    # See the images
    cv2.imshow('Real capture', np.hstack([frameL, frameR]))

    # undistort
    dstR = cv2.undistort(frameR, mtxR, distR, None, newcameramtxR)
    dstL = cv2.undistort(frameL, mtxL, distL, None, newcameramtxL)
    cv2.imshow('Rectidifed images', np.hstack([dstL, dstR]))

    # If key is 'c' for capture
    if key & 0xFF == ord('q'):
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
