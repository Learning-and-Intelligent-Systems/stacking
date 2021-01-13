import itertools
import json
import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco
import os

from rotation_util import *


# here are some calibration parameters for Realsense D435 that I found online
# in the future we will need to calibrate the cameras with a grid
resolution_x = 640
resolution_y = 360
fx = 322.282
fy = 322.282
cx = resolution_x/2
cy = resolution_y/2
mtx = np.array([[fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]])
dist = np.zeros(5) # no distortion

# camera extrinsics (4x4 pose matrix)
X_WC = Rt_to_pose_matrix(eul_to_rot([-np.pi/2, 0, 0]), [0, -1, 0])

# create the aruco dictionary and default parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params =  aruco.DetectorParameters_create()

# 8 x 3 matrix of -1, 1 to compute the corners of the blocks
signed_corners = np.array([c for c in itertools.product([-1, 1], repeat=3)])

def get_block_info():
    """ load all the block info files from the tags/ folder

    block info files are created by create_aruco_texture_for_block.
    They contain information about the aruco tags associated with the block
    and the poses of those tags in the block frame
    
    Returns:
        json -- {block_id: block_info}
    """
    block_info = {}
    for fname in os.listdir('tags'):
        print(fname)
        if fname.endswith('info.json'):
            block_id = int(fname.split('_')[1])
            print(block_id)
            with open('tags/' + fname, 'r') as f:
                block_info[block_id] = json.load(f)

    return block_info


def draw_object(X_CO, dimensions, frame):
    """[summary]
    
    [description]
    
    Arguments:
        X_CO {np.ndarray} -- 4x4 pose of the object in the camera frame
        dimensions {np.ndarray} -- x,y,z dimensions of the object
    """
    # compute the object pose in world frame
    X_WO = X_WC @ X_CO
    # get the translation from the COG to the box corner points
    t_OP = signed_corners * dimensions[None,:] / 2
    # and convert the corner points to world frame
    t_WP = np.array([(X_CO @ Rt_to_pose_matrix(np.eye(3), t_OP_i))[:3,3] for t_OP_i in t_OP])

    R_WC, t_WC = pose_matrix_to_Rt(X_WC)

    image_points, _ = cv2.projectPoints(t_WP, R_WC, t_WC, mtx, dist)
    for pt in image_points:
        cv2.circle(frame, tuple(pt[0].astype(int)), 5, (255,255,255), -1)



def main():
    block_info = get_block_info()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    theta = 0

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
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
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
                    # print(rmat, tvec[i])
                    # draw axis for the aruco markers
                    aruco.drawAxis(color_image, mtx, dist, rvec[i], tvec[i], 0.05)

                    # draw the detected object into the frame
                    draw_object(Rt_to_pose_matrix(eul_to_rot((theta, theta, theta)),
                                np.zeros(3)), np.array([6, 10, 6])/100, color_image)
                    theta += np.pi/100

                # color_image = aruco.drawDetectedMarkers(
                #     color_image.copy(), corners, ids)

            # # Show images
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Aruco Frames', color_image)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        pipeline.stop()

if __name__ == '__main__':
    main()
