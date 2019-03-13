import numpy as np
import os

# Files for calibration
filenameL = os.path.join("models/", "{}.npy".format("imgpointsL"))
filenameR = os.path.join("models/", "{}.npy".format("imgpointsR"))
filename_op = os.path.join("models/", "{}.npy".format("objpoints"))
filename_mtR = os.path.join("models/", "{}.npy".format("mtxR"))
filename_dR = os.path.join("models/", "{}.npy".format("distR"))
filename_mtL = os.path.join("models/", "{}.npy".format("mtxL"))
filename_dL = os.path.join("models/", "{}.npy".format("distL"))
filename_chR = os.path.join("models/", "{}.npy".format("ChessImaR"))


def write_calibration(imgpointsL, imgpointsR, objpoints, mtxR, distR, mtxL, distL, ChessImaR):
    np.save(filenameL, imgpointsL)
    np.save(filenameR, imgpointsR)
    np.save(filename_op, objpoints)
    np.save(filename_mtR, mtxR)
    np.save(filename_dR, distR)
    np.save(filename_mtL, mtxL)
    np.save(filename_dL, distL)
    np.save(filename_chR, ChessImaR)

def read_calibration():
    imgpointsR = np.load(filenameR)
    imgpointsL = np.load(filenameL)
    objpoints = np.load(filename_op)
    mtxR = np.load(filename_mtR)
    distR = np.load(filename_dR)
    mtxL = np.load(filename_mtL)
    distL = np.load(filename_dL)
    ChessImaR = np.load(filename_chR)

    return imgpointsR, imgpointsL, objpoints, mtxR, distR, mtxL, distL, ChessImaR
