import cv2
import glob

# Variables for calibration
num_pictures = 40
chessboard_dim = (8, 5)
pic = 0

# Image variables
ChessImaR = None
ChessImaL = None

# Termination criteria
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Call the two cameras
input("Connect the right camera and press any key ...")
devices = glob.glob("/dev/video?")
CamR = cv2.VideoCapture(devices[0])

input("Connect the left camera and press any key ...")
devices = glob.glob("/dev/video?")
CamL = cv2.VideoCapture(devices[0])


# Start the calibration process
print("Calibration process started, press 'C' to take a picture")

while True:

    key = cv2.waitKey(1)

    # Capture the frame
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    # Convert to gray scale
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

    # See the images
    cv2.imshow('imgR', cv2.resize(frameR, (320, 240)))
    cv2.imshow('imgL', cv2.resize(frameL, (320, 240)))

    # If key is 'c' for capture
    if key & 0xFF == ord('c'):

        # Find the chess board corners
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_dim, None)
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_dim, None)

        # If found, add object points, image points (after refining them)
        if retR and retL:

            corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria_stereo)
            corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria_stereo)

            # Draw and display the corners
            cv2.drawChessboardCorners(grayR, chessboard_dim, corners2R, retR)
            cv2.drawChessboardCorners(grayL, chessboard_dim, corners2L, retL)

            cv2.imshow('VideoR', cv2.resize(grayR, (320, 240)))
            cv2.imshow('VideoL', cv2.resize(grayL, (320, 240)))

            # Save the image in the file where this Programm is located
            cv2.imwrite('images/chessboard-R'+str(pic)+'.png',frameR)
            cv2.imwrite('images/chessboard-L'+str(pic)+'.png',frameL)

            pic = pic + 1
            print('Chessboard {0}/{1}'.format(pic, num_pictures))

        # End the Programme
        if pic > num_pictures:
            print("Chessboard capturing done! Please run the calibration program")
            break

    # If key is 'q' break
    if key & 0xFF == ord('q'):
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
