# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:22:50 2019

@author: harshak
"""
import glob
import numpy as np
import cv2 as cv

mtx=np.loadtxt('C:/Users/harshak/Documents/GitHub/New Folder With Items/rm.txt', delimiter=',')
#print(mtx)
dist=np.loadtxt('C:/Users/harshak/Documents/GitHub/New Folder With Items/rd.txt', delimiter=',')
#print(dist)
images = glob.glob('U3TL0002*.jpg')
for fname in images:
    img = cv.imread(fname)

#print(img.shape)
    h,  w = img.shape[:2]
#newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


# undistort
    dst = cv.undistort(img, mtx, dist, None,mtx)
# crop the image

#x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
    cv.imwrite('C:/Users/harshak/Documents/GitHub/TouchLess2019/U3TL0002/Undistorted/positive y/'+fname, dst)
