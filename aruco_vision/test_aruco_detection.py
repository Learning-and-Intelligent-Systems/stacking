""" Massachusetts Institute of Technology

Izzybrand, 2020
"""
import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco

from cal import mtx, dist

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# create the aruco dictionary and default parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params =  aruco.DetectorParameters_create()

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # convert image to numpy array
        color_image = np.asarray(color_frame.get_data())
        # convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # detect aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params)

        # if we've detected markers, then estimate their pose and draw frames
        if ids is not None:
            # NOTE(izzy): in the future we will need to look up the marker
            # length based on the marker id
            marker_length = 0.05
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(
                corners, marker_length, mtx, dist)
            for i in range(0, ids.size):
                # this is the pose of the marker in the camera frame
                rmat, jacobian = cv2.Rodrigues(rvec[i])
                print(rmat, tvec[i])
                # draw axis for the aruco markers
                aruco.drawAxis(color_image, mtx, dist, rvec[i], tvec[i], 0.05)

            # color_image = aruco.drawDetectedMarkers(
            #     color_image.copy(), corners, ids)

        # # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Aruco Frames', color_image)
        cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()
