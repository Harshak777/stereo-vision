# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:56:55 2019

@author: harshak
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
print('hi')
imgL = cv2.imread('C:/Users/harshak/Desktop/3.jpg',0)
imgR = cv2.imread('C:/Users/harshak/Desktop/4.jpg',0)
print('done')
stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(16)
stereoMatcher.setNumDisparities(96)
stereoMatcher.setBlockSize(11)
stereoMatcher.setSpeckleRange(35)
stereoMatcher.setSpeckleWindowSize(100)
disparity = stereoMatcher.compute(imgL, imgR)

print('here')
norm_image = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

'''cv2.imshow('norm_image', norm_image)
cv2.imwrite('C:/Users/harshak/Documents/GitHub/depth2.png',disparity)'''
plt.imshow(disparity,'gray')
plt.show()