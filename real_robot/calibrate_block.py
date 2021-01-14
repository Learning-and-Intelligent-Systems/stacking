""" Massachusetts Institute of Technology

Izzybrand, 2020
"""
import itertools
import pickle
import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco
import sys

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

# create the aruco dictionary and default parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params =  aruco.DetectorParameters_create()

# this list provides info about each face of the object
# R_CO is the rotation matrix of the object in the camera frame that
# corresponds to the text description.
# t_OT_dimensions_coeff is a basis vector for the tag that is facing the
# camera, but in the object frame
face_infos = [
    {
        'face_toward_camera': 'FRONT',
        'face_up': 'TOP',
        'R_CO': eul_to_rot([np.pi/2, 0, 0]),
        't_OT_dimensions_coeff': np.array([0, -1., 0])
    },
    {
        'face_toward_camera': 'BACK',
        'face_up': 'TOP',
        'R_CO': eul_to_rot([-np.pi/2, 0, np.pi]),
        't_OT_dimensions_coeff': np.array([0, 1., 0])
    },
    {
        'face_toward_camera': 'RIGHT',
        'face_up': 'TOP',
        'R_CO': eul_to_rot([np.pi/2, np.pi/2, 0]),
        't_OT_dimensions_coeff': np.array([1., 0, 0])
    },
    {
        'face_toward_camera': 'LEFT',
        'face_up': 'TOP',
        'R_CO': eul_to_rot([np.pi/2, -np.pi/2, 0]),
        't_OT_dimensions_coeff': np.array([-1., 0, 0])
    },
    {
        'face_toward_camera': 'TOP',
        'face_up': 'FRONT',
        'R_CO': eul_to_rot([0,np.pi, 0]),
        't_OT_dimensions_coeff': np.array([0, 0, 1.])
    },
    {
        'face_toward_camera': 'BOTTOM',
        'face_up': 'FRONT',
        'R_CO': eul_to_rot([0,0, 0]),
        't_OT_dimensions_coeff': np.array([0, 0, -1.])
    }
]

def detect_aruco(color_image):
    """ detect an the aruco markers in a color image. return None
    if None found.

    Arguments:
        color_image {np.ndarray} -- a frame of video
    """
    # convert to grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    # detect aruco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray_image, aruco_dict, parameters=aruco_params)

    if ids is None:
        return None
    else:
        return corners, ids

def detect_block_id(pipeline):
    """ pull and display images from the pipeline until a single aruco tag is
    visible in the frame. At that point, compute the block id by tag_id // 6
    and return

    Arguments:
        pipeline {rs.pipeline} -- the stream from the realsense

    Returns:
        int -- the id of the observed block
    """
    block_id = None
    while block_id is None:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        # convert image to numpy array
        color_image = np.asarray(color_frame.get_data())
        # look for aruco tags in the frames
        result = detect_aruco(color_image)
        if result is not None:
            corners, ids = result
            # display the frame with the markers drawn in
            color_image = aruco.drawDetectedMarkers(
                    color_image.copy(), corners, ids)

            # if we get a single tag visible, that is the block we're calibrating
            if ids.size == 1:
                block_id = ids.item() // 6
                print(f'Got it! I saw block id {block_id}')
            else:
                print('Multiple markers visible')

        cv2.imshow('Aruco Frames', color_image)
        cv2.waitKey(1)

    return block_id

def detect_face(pipeline, face_info, info):
    """ pull and display images from the pipeline until a single aruco tag is
    visible in the frame. At that point, compute the rotation of the tag in the
    expected frame of the object (specified in face_info['R_CO']). Then snap
    the observed object frame rotation to an axis and save it in the info dict

    Arguments:
        pipeline {rs.pipeline} -- the stream from the realsense
        face_info {dict} -- contains information about the current face (like the pose)
        info {dict} -- contains information about the block

    Side Effects:
        Adds fields to info
    """
    tag_id = None
    while tag_id is None:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        # convert image to numpy array
        color_image = np.asarray(color_frame.get_data())
        # look for aruco tags in the frames
        result = detect_aruco(color_image)
        if result is not None:
            corners, ids = result
            # draw a reference axis for the block
            aruco.drawAxis(color_image, mtx, dist, face_info['R_CO'], np.array([0,0,0.2]), 0.1)
            # if we've detected a marker, then estimate their pose and draw frames
            color_image = aruco.drawDetectedMarkers(
                color_image.copy(), corners, ids)
            # if we've got a single tag
            if ids.size == 1:
                tag_id = ids.item()
                # get the pose of the gae in the camera frame
                marker_length = info[tag_id]['marker_size_cm']/100.
                rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(
                    corners, marker_length, mtx, dist)
                # compute the pose of the tag in the object frame, snapped to an axis
                R_CT = cv2.Rodrigues(rvec)[0] # rotation of tag in camera frame
                R_OT = np.linalg.inv(face_info['R_CO']) @ R_CT # rotation of tag in object frame
                snapped_R_OT = snap_rotation_matrix(R_OT) # snap that rotation to an axis
                snapped_R_CT = face_info['R_CO'] @ snapped_R_OT # and compute the snapped tag rotation

                # update the tag in info
                t_OT = face_info['t_OT_dimensions_coeff'] * info['dimensions'] / 2.
                X_OT = Rt_to_pose_matrix(snapped_R_OT, t_OT)
                info[tag_id]['X_OT'] = X_OT
                info[tag_id]['name'] = face_info['face_toward_camera']

                # draw the pose of the tag before and after snapping
                aruco.drawAxis(color_image, mtx, dist, R_CT, tvec, marker_length)
                aruco.drawAxis(color_image, mtx, dist, snapped_R_CT, tvec, marker_length)
                cv2.imshow('Aruco Frames', color_image)
                cv2.waitKey(1)

                print(f'Got it! I saw tag id {tag_id}')
            else:
                print('Multiple markers visible')

        cv2.imshow('Aruco Frames', color_image)
        cv2.waitKey(1)

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        # detect the block to be calibrated by having the user place the block
        # front of the camera
        print('Place the block to be calibrated in front of the camera.')
        input('Press enter when ready')
        block_id = detect_block_id(pipeline)

        # open the info file for the block we just recognized
        try:
            with open(f'tags/block_{block_id}_info.pkl', 'rb') as f:
                info = pickle.load(f)
        except FileNotFoundError:
            print(f'Failed to find the info file for block {block_id}.',
                   'Did you forget to run create_aruco_block.py?')
            sys.exit(1)

        # Now calibrate each face of the block
        for face_info in face_infos:
            print(f'Place the block with {face_info["face_toward_camera"]}',
                  f'toward the camera and {face_info["face_up"]} up')
            input('Press enter when ready')
            detect_face(pipeline, face_info, info)

    finally:
        # Stop streaming
        pipeline.stop()

    # save the resulting calibration file
    info_filename = f'tags/block_{block_id}_info.pkl'
    with open(info_filename, 'wb') as f:
        pickle.dump(info, f)

    print(f'Saved block info to {info_filename}')


if __name__ == '__main__':
    main()
