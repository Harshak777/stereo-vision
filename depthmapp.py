# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:21:59 2019

@author: harshak
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

print('hi')
'''imgL = cv2.imread('C:/Users/harshak/Desktop/3.jpg',0)
imgR = cv2.imread('C:/Users/harshak/Documents/GitHub/U3TL0002_001_5d.jpg',0)'''

imgL = cv2.pyrDown(cv2.imread('C:/Users/harshak/Documents/GitHub/U3TL0001_001_0d.jpg'))  # downscale images for faster processing
imgR = cv2.pyrDown(cv2.imread('C:/Users/harshak/Documents/GitHub/U3TL0002_001_0d.jpg'))

print('done')

window_size = 5
min_disp = 16
num_disp = 144 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 30,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 10,
        speckleRange = 32
    )

print('computing disparity...')
disp = stereo.compute(imgL, imgR).astype(np.float32)


#write_ply('out.ply', out_points, out_colors)
#print('%s saved' % 'out.ply')

#cv2.imshow('left', imgL)
#cv2.imshow('disparity', (disp-min_disp)/num_disp)
#cv2.waitKey()

print('Done')

plt.imshow((disp-min_disp)/num_disp,'gray')
plt.show()

#cv2.imshow('disparity', (disp-min_disp)/num_disp)