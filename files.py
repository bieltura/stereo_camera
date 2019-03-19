import numpy as np
import os

# Files for calibration - stereo
filenameL = os.path.join("models/stereo", "{}.npy".format("imgpointsL"))
filenameR = os.path.join("models/stereo", "{}.npy".format("imgpointsR"))
filename_op = os.path.join("models/stereo", "{}.npy".format("objpoints"))
filename_mtR = os.path.join("models/stereo", "{}.npy".format("mtxR"))
filename_dR = os.path.join("models/stereo", "{}.npy".format("distR"))
filename_mtL = os.path.join("models/stereo", "{}.npy".format("mtxL"))
filename_dL = os.path.join("models/stereo", "{}.npy".format("distL"))
filename_chR = os.path.join("models/stereo", "{}.npy".format("ChessImaR"))

# Files for calibration - single
filename = os.path.join("models/single", "{}.npy".format("imgpoints"))
filename_ops = os.path.join("models/single", "{}.npy".format("objpoints"))
filename_mt = os.path.join("models/single", "{}.npy".format("mtx"))
filename_d = os.path.join("models/single", "{}.npy".format("dist"))
filename_ch = os.path.join("models/single", "{}.npy".format("ChessIma"))


def write_stereo_calibration(imgpointsL, imgpointsR, objpoints, mtxR, distR, mtxL, distL, ChessImaR):
    np.save(filenameL, imgpointsL)
    np.save(filenameR, imgpointsR)
    np.save(filename_op, objpoints)
    np.save(filename_mtR, mtxR)
    np.save(filename_dR, distR)
    np.save(filename_mtL, mtxL)
    np.save(filename_dL, distL)
    np.save(filename_chR, ChessImaR)


def read_stereo_calibration():
    imgpointsR = np.load(filenameR)
    imgpointsL = np.load(filenameL)
    objpoints = np.load(filename_op)
    mtxR = np.load(filename_mtR)
    distR = np.load(filename_dR)
    mtxL = np.load(filename_mtL)
    distL = np.load(filename_dL)
    ChessImaR = np.load(filename_chR)

    return imgpointsR, imgpointsL, objpoints, mtxR, distR, mtxL, distL, ChessImaR


def write_single_calibration(imgpoints, objpoints, mtx, dist, ChessIma):
    np.save(filename, imgpoints)
    np.save(filename_ops, objpoints)
    np.save(filename_mt, mtx)
    np.save(filename_d, dist)
    np.save(filename_ch, ChessIma)


def read_single_calibration():
    imgpoints = np.load(filename)
    objpoints = np.load(filename_ops)
    mtx = np.load(filename_mt)
    dist = np.load(filename_d)
    ChessIma = np.load(filename_ch)

    return imgpoints, objpoints, mtx, dist, ChessIma
