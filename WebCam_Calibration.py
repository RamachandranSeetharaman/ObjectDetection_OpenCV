# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:57:08 2020
Calibration of Webcamera

@author: Sriram-PC
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np


left = cv2.VideoCapture(1)  #  Primary Webcam
right = cv2.VideoCapture(0) # Secondary Webcam
i=0
_3d_points=[]
_2d_points=[]
_3d_corners = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                           [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])

while(i<14):
    ret, L_frame = left.read()
    ret, R_frame = right.read()
    #cv2.imshow('left', L_frame)
    #cv2.imshow('right', R_frame)
    print(L_frame.shape)
    print(R_frame.shape)
    plt.subplot(121)
    plt.imshow(R_frame[...,::-1])
    plt.subplot(122)
    plt.imshow(L_frame[...,::-1])
    plt.show()
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break
        
    ret, corners = cv2.findChessboardCorners(L_frame, (7,6))
    print(corners.shape)
    print(corners.shape[0])
    corners=corners.reshape(-1,2)
    print(corners.shape)
    print(corners[0])
    L_frame_vis=L_frame.copy()
    cv2.drawChessboardCorners(L_frame_vis, (7,6), corners, ret) 
    plt.imshow(L_frame_vis)
    plt.show()
    
    x,y=np.meshgrid(range(7),range(6))
    print("x:\n",x)
    print("y:\n",y)
    world_points=np.hstack((x.reshape(42,1),y.reshape(42,1),np.zeros((42,1)))).astype(np.float32)
    print(world_points)
    print(corners[0],'->',world_points[0])
    print(corners[35],'->',world_points[35])
    
        
    if ret: #add points only if checkerboard was correctly detected:
         _2d_points.append(corners) #append current 2D points
         _3d_points.append(world_points) #3D points are always the same
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(_3d_points, _2d_points, (L_frame.shape[1],L_frame.shape[0]),None, None)
    print("Ret:",ret)
    print("Mtx:",mtx," ----------------------------------> [",mtx.shape,"]")
    print("Dist:",dist," ----------> [",dist.shape,"]")
    print("rvecs:",rvecs," --------------------------------------------------------> [",rvecs[0].shape,"]")
    print("tvecs:",tvecs," -------------------------------------------------------> [",tvecs[0].shape,"]")
    L_frame_undistorted=cv2.undistort(L_frame, mtx, dist)
    plt.subplot(121)
    plt.imshow(L_frame)
    plt.subplot(122)
    plt.imshow(L_frame_undistorted)
    plt.show()
    
    i+=1
left.release()
right.release()
cv2.destroyAllWindows()


