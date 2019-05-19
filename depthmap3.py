# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:55:18 2019

@author: harshak
"""
import cv2 as cv2
from matplotlib import pyplot as plt

img_1_downsampled=cv2.imread('C:/Users/harshak/Documents/GitHub/U3TL0001_001_5d.jpg',0)
img_2_downsampled=cv2.imread('C:/Users/harshak/Documents/GitHub/U3TL0002_001_5d.jpg',0)

def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image

#Downsample each image 3 times (because they're too big)
#img_1_downsampled = downsample_image(img_1_undistorted,1)
#img_2_downsampled = downsample_image(img_2_undistorted,1)
#Set disparity parameters
#Note: disparity range is tuned according to specific parameters obtained through trial and error. 
win_size = 5
min_disp = 48
max_disp = 432 #min_disp * 9
num_disp = max_disp - min_disp # Needs to be divisible by 16
#Create Block matching object. 
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
 numDisparities = num_disp,
 blockSize = 15,
 uniquenessRatio = 10,
 speckleWindowSize = 10,
 speckleRange = 10,
 disp12MaxDiff = 1,
 P1 = 8*3*win_size**2,#8*3*win_size**2,
 P2 =32*3*win_size**2) #32*3*win_size**2)
#Compute disparity map
print ("\nComputing the disparity  map...")
disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)

norm_image = cv2.normalize(disparity_map, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.imwrite('C:/Users/harshak/Documents/GitHub/depth3.png',norm_image)
#Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
plt.imshow(disparity_map,'gray')
plt.show()