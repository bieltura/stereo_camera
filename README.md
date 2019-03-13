# Stereo cameras
Program for calibrating two stereo cameras to generate depth information.
Packages required:
  - OpenCV
  - Numpy
  - Glob

# How-to
calibrate.py generates the images and calibration files inside images/ and models/ folder.
stereo.py loads the calibration models and creates a stereo depth map from this calibration.
To change the chessboard pattern take a look at the variable chessboard_dim inside calibration.py

# Credits
Stereo cameras uses a number of open source projects to work properly:

* [LearnTechWithUs] - Basic stereo code
* [DavidCastillo] - Improved version of stereo code
* [OpenCV] - Amazing tutorialsMarkdown parser done right. Fast and easy to extend.
