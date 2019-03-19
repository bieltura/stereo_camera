import cv2
import numpy as np
import files

# Variables for calibration
num_pictures = 40
chessboard_dim = (9, 6)
pic = 0

# Image variables
ChessImaR = None

# Prepare object points (index of chessboard points)
objp = np.zeros((chessboard_dim[0] * chessboard_dim[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_dim[0], 0:chessboard_dim[1]].T.reshape(-1, 2)

# Termination criteria
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all images
objpoints = []   # 3d points in real world space
imgpoints = []

print("Files for calibration generated")
print("Loading chessboard pictures")

for pic in range(num_pictures):

    # Capture the frame
    frame = cv2.imread('images_single/chessboard-single'+str(pic)+'.png')

    # Convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_dim, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria_stereo)
        imgpoints.append(corners)

        print('Loaded chessboard {0}/{1}'.format(pic, num_pictures))

# Calibration
_, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Right camera: A matrix: \n{}".format(mtx))
print("Distoriton coefficients: \n{}".format(dist))

files.write_single_calibration(imgpoints, objpoints, mtx, dist, gray)

print("Calibration ended")
