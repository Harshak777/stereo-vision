# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:00:58 2019

@author: harshak
"""

#import sys
import numpy as np
import cv2

REMAP_INTERPOLATION = cv2.INTER_LINEAR

DEPTH_VISUALIZATION_SCALE = 2048


calibration = np.load('C:/Users/harshak/Documents/GitHub/det.npz', allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# TODO: Use more stable identifiers
left = cv2.imread('C:/Users/harshak/Documents/GitHub/U3TL0001_001_5d.jpg')
right = cv2.imread('C:/Users/harshak/Documents/GitHub/U3TL0002_001_5d.jpg')

# Increase the resolution


# The distortion in the left and right edges prevents a good calibration, so
# discard the edges
CROP_WIDTH = 960
def cropHorizontal(image):
    return image[:,
            int((CAMERA_WIDTH-CROP_WIDTH)/2):
            int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]

# TODO: Why these values in particular?
# TODO: Try applying brightness/contrast/gamma adjustments to the images
stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(4)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setBlockSize(21)
stereoMatcher.setROI1(leftROI)
stereoMatcher.setROI2(rightROI)
stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(45)

# Grab both frames first, then retrieve to minimize latency between cameras


leftHeight, leftWidth = left.shape[:2]

rightHeight, rightWidth = right.shape[:2]


fixedLeft = cv2.remap(left, leftMapX, leftMapY, REMAP_INTERPOLATION)
fixedRight = cv2.remap(right, rightMapX, rightMapY, REMAP_INTERPOLATION)

grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
depth = stereoMatcher.compute(grayLeft, grayRight)

cv2.imshow('left', fixedLeft)
cv2.imshow('right', fixedRight)
cv2.imwrite('C:/Users/harshak/Documents/GitHub/depth.jpg', depth)
cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
    
