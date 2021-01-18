""" Massachusetts Institute of Technology

Izzybrand, 2020
"""
import itertools
import pickle
import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco
import os

from cal import dist, mtx
from rotation_util import *

# create the aruco dictionary and default parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params =  aruco.DetectorParameters_create()

# 8 x 3 matrix of -1, 1 to compute the corners of the blocks (used in draw_block)
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
        if fname.endswith('info.pkl'):
            block_id = int(fname.split('_')[1])
            with open('tags/' + fname, 'rb') as f:
                block_info[block_id] = pickle.load(f)

    return block_info

def draw_block(X_CO, dimensions, color_image, draw_axis=True):
    """ Draw dots at the corners of the block at the specified camera-frame
    pose and with the given dimensions.

    The top corners are lighter in color and the bottom corners are darker.

    Arguments:
        X_CO {np.ndarray} -- 4x4 pose of the object in the camera frame
        dimensions {np.ndarray} -- x,y,z dimensions of the object
        color_image {np.ndarray} -- the image into which to draw the object

    Optional
        draw_axis {bool} -- Draw the block axis at the COG. Default True

    Side effects:
        Modifies color_image
    """
    # get the translation from the COG to the box corner points
    t_OP = signed_corners * dimensions[None,:] / 2
    # and convert the corner points to camera frame
    t_CP = np.array([(X_CO @ Rt_to_pose_matrix(np.eye(3), t_OP_i))[:3,3] for t_OP_i in t_OP])
    # project the points into the image coordinates
    image_points, _ = cv2.projectPoints(t_CP, np.eye(3), np.zeros(3), mtx, dist)
    # and draw them into the image
    for corner, image_pt in zip(signed_corners, image_points):
        color = np.array([1.0,0.0,0.7])*100 + (corner[2] > 0) * 155
        cv2.circle(color_image, tuple(image_pt[0].astype(int)), 5, color, -1)

    if draw_axis:
        R_CO, t_CO = pose_matrix_to_Rt(X_CO)
        aruco.drawAxis(color_image, mtx, dist, R_CO, t_CO, dimensions.min()/2)

def get_block_poses_in_camera_frame(ids, corners, info, color_image=None):
    tag_id_to_block_pose = {}
    for i in range(0, ids.size):
        # pull out the info corresponding to this block
        tag_id = ids[i][0]
        block_id = tag_id // 6
        try:
            marker_info = info[block_id][tag_id]
            print(f'Detected {tag_id}, the {marker_info["name"]} face of block {block_id}')
        except KeyError:
            print('Failed to find the block info for {block_id}. Skipping.')
            continue
        # detect the aruco tag
        marker_length = marker_info["marker_size_cm"]/100. # in meters
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(
            corners[i], marker_length, mtx, dist)
        # pose of the tag in the camera frame
        X_CT = Rt_to_pose_matrix(cv2.Rodrigues(rvec)[0], tvec)
        # pose of the object in camera frame
        X_TO = np.linalg.inv(marker_info["X_OT"])
        X_CO = X_CT @ X_TO
        # # draw axis for the aruco markers
        tag_id_to_block_pose[tag_id] = X_CO

        if color_image is not None:
            aruco.drawAxis(color_image, mtx, dist, rvec, tvec, marker_length/2)

    return tag_id_to_block_pose

def combine_block_poses(block_poses_in_camera_frame):
    # consolidate all the tags for each block into a list indexed by the block_id
    block_poses = {}
    for tag_id in block_poses_in_camera_frame.keys():
        block_id = tag_id // 6
        if block_id not in block_poses:
            block_poses[block_id] = []
        block_poses[block_id].append(block_poses_in_camera_frame[tag_id])

    for block_id in block_poses.keys():
        poses = block_poses[block_id]
        block_poses[block_id] = mean_pose(poses)

    return block_poses

def main():
    info = get_block_info()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

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
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            # detect aruco markers
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                gray_image, aruco_dict, parameters=aruco_params)

            # if we've detected markers, then estimate their pose and draw frames
            if ids is not None:
                # estimate the pose of the blocks (one for each visible tag)
                tag_id_to_block_pose = \
                    get_block_poses_in_camera_frame(ids, corners, info)

                # draw a block for each detected tag (to show disagreement)
                # for tag_id in tag_id_to_block_pose.keys():
                #     block_id = tag_id // 6
                #     X_CO = tag_id_to_block_pose[tag_id]
                #     dimensions = info[block_id]['dimensions']
                #     draw_block(X_CO, dimensions, color_image)

                # combine all the visible tags
                block_id_to_block_pose = combine_block_poses(tag_id_to_block_pose)
                for block_id in block_id_to_block_pose.keys():
                    X_CO = block_id_to_block_pose[block_id]
                    dimensions = info[block_id]['dimensions']
                    draw_block(X_CO, dimensions, color_image)


            cv2.imshow('Aruco Frames', color_image)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == '__main__':
    main()
