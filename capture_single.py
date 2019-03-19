import cv2
import glob

# Variables for calibration
num_pictures = 40
chessboard_dim = (9, 6)
pic = 0

# Image variables
ChessImaR = None
ChessImaL = None

# Termination criteria
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Call the two cameras
input("Connect the camera and press any key ...")
devices = glob.glob("/dev/video?")
Cam = cv2.VideoCapture(0)

# Start the calibration process
print("Calibration process started, press 'C' to take a picture")

while True:

    key = cv2.waitKey(1)

    # Capture the frame
    ret, frame = Cam.read()

    if frame is not None:

        # Convert to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # See the images
        cv2.imshow('imgR', cv2.resize(frame, (320, 240)))

        # If key is 'c' for capture
        if key & 0xFF == ord('c'):

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_dim, None)

            # If found, add object points, image points (after refining them)
            if ret:

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria_stereo)

                # Draw and display the corners
                cv2.drawChessboardCorners(gray, chessboard_dim, corners2, ret)

                cv2.imshow('VideoR', cv2.resize(gray, (320, 240)))

                # Save the image in the file where this Programm is located
                cv2.imwrite('images_single/chessboard-single'+str(pic)+'.png', frame)

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
Cam.release()
cv2.destroyAllWindows()
