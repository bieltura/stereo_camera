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

wR = int(CamR.get(cv2.CAP_PROP_FRAME_WIDTH))
hR = int(CamR.get(cv2.CAP_PROP_FRAME_HEIGHT))
newcameramtxR, roi = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 0, (wR, hR))

input("Connect the left camera and press any key ...")
devices = glob.glob("/dev/video?")
CamL = cv2.VideoCapture(devices[0])

wL = int(CamL.get(cv2.CAP_PROP_FRAME_WIDTH))
hL = int(CamL.get(cv2.CAP_PROP_FRAME_HEIGHT))
newcameramtxL, roi = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 0, (wL, hL))

while True:

    key = cv2.waitKey(1)

    # Capture the frame
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    # See the images
    cv2.imshow('imgR', cv2.resize(frameR, (480, 320)))
    cv2.imshow('imgL', cv2.resize(frameL, (480, 320)))

    # undistort
    dstR = cv2.undistort(frameR, mtxR, distR, None, newcameramtxR)
    dstL = cv2.undistort(frameL, mtxL, distL, None, newcameramtxL)
    cv2.imshow('imgRU', cv2.resize(dstR, (480, 320)))
    cv2.imshow('imgLU', cv2.resize(dstL, (480, 320)))

    # If key is 'c' for capture
    if key & 0xFF == ord('q'):
        break


# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
