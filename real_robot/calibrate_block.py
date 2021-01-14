import itertools
import pickle
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
# X_WC = Rt_to_pose_matrix(eul_to_rot([-np.pi/2, 0, 0]), [0, 0, 0])
X_WC = np.eye(4)

# create the aruco dictionary and default parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params =  aruco.DetectorParameters_create()

# 8 x 3 matrix of -1, 1 to compute the corners of the blocks
signed_corners = np.array([c for c in itertools.product([-1, 1], repeat=3)])


face_infos = [
    {
        'face_toward_camera': 'front',
        'face_up': 'top',
        'R_CO': eul_to_rot([np.pi/2, 0, 0]),
        't_OT_dimensions_coeff': np.array([0, -0.5, 0])
    },
    {
        'face_toward_camera': 'back',
        'face_up': 'top',
        'R_CO': eul_to_rot([-np.pi/2, 0, np.pi]),
        't_OT_dimensions_coeff': np.array([0, 0.5, 0])
    },
    {
        'face_toward_camera': 'right',
        'face_up': 'top',
        'R_CO': eul_to_rot([np.pi/2, np.pi/2, 0]),
        't_OT_dimensions_coeff': np.array([0.5, 0, 0])
    },
    {
        'face_toward_camera': 'left',
        'face_up': 'top',
        'R_CO': eul_to_rot([np.pi/2, -np.pi/2, 0]),
        't_OT_dimensions_coeff': np.array([-0.5, 0, 0])
    },
    {
        'face_toward_camera': 'top',
        'face_up': 'front',
        'R_CO': eul_to_rot([0,np.pi, 0]),
        't_OT_dimensions_coeff': np.array([0, 0, 0.5])
    },
    {
        'face_toward_camera': 'bottom',
        'face_up': 'front',
        'R_CO': eul_to_rot([0,0, 0]),
        't_OT_dimensions_coeff': np.array([0, 0, -0.5])
    }
]

def draw_block(X_CO, dimensions, color_image):
    """[summary]
    
    [description]
    
    Arguments:
        X_CO {np.ndarray} -- 4x4 pose of the object in the camera frame
        dimensions {np.ndarray} -- x,y,z dimensions of the object
    """
    R_WC, t_WC = pose_matrix_to_Rt(X_WC)
    # compute the object pose in world frame
    X_WO = X_WC @ X_CO
    # get the translation from the COG to the box corner points
    t_OP = signed_corners * dimensions[None,:] / 2
    # and convert the corner points to world frame
    t_WP = np.array([(X_CO @ Rt_to_pose_matrix(np.eye(3), t_OP_i))[:3,3] for t_OP_i in t_OP])
    # project the points into the camera frame
    image_points, _ = cv2.projectPoints(t_WP, R_WC, t_WC, mtx, dist)
    # and draw them into the image
    for corner, image_pt in zip(signed_corners, image_points):
        color = (corner > 0) * 255.0
        cv2.circle(color_image, tuple(image_pt[0].astype(int)), 5, color, -1)


    R_WO, t_WO = pose_matrix_to_Rt(X_WO)
    aruco.drawAxis(color_image, mtx, dist, R_WO, t_WO, 0.05)


def main():
    block_id = 4
    with open(f'tags/block_{block_id}_info.pkl', 'rb') as f:
        info = pickle.load(f)

    dimensions = info['dimensions']

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    

    try:
        for face_info in face_infos:
            print(f'Place the block with {face_info["face_toward_camera"]} toward the camera and {face_info["face_up"]} up')
            R_CO = face_info["R_CO"]
            X_CO = Rt_to_pose_matrix(R_CO, [0,0,0.2])
            input('Press enter when ready')

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
                aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
                # detect aruco markers
                corners, ids, rejectedImgPoints = aruco.detectMarkers(
                    gray_image, aruco_dict, parameters=aruco_params)


                draw_block(X_CO, dimensions, color_image)
                # if we've detected a marker, then estimate their pose and draw frames
                if ids is not None:
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
                        R_CT = cv2.Rodrigues(rvec)[0]
                        R_OT = np.linalg.inv(face_info['R_CO']) @ R_CT
                        snapped_R_CT = snap_rotation_matrix(R_CT)
                        snapped_R_OT = snap_rotation_matrix(R_OT)

                        # update the tag info in the json
                        t_OT = face_info['t_OT_dimensions_coeff'] * dimensions
                        X_OT = Rt_to_pose_matrix(snapped_R_OT, t_OT)
                        info[tag_id]['X_OT'] = X_OT
                        info[tag_id]['name'] = face_info['face_toward_camera']

                        # draw the resulting snapped tag
                        aruco.drawAxis(color_image, mtx, dist, snapped_R_CT, tvec, marker_length)
                        cv2.imshow('Aruco Frames', color_image)
                        cv2.waitKey(1)

                        print(f'Got it! I saw tag id {tag_id}')
                        break
                    else:
                        print('Multiple markers visible')

                    # draw the detected object into the frame
                    # X_CO = Rt_to_pose_matrix(eul_to_rot((theta, theta, theta)), np.array([0,0,0.2]))
                    # draw_block(X_CO, np.array([0.06, 0.1, 0.06]), color_image)
                    # theta += np.pi/100

                    # color_image = aruco.drawDetectedMarkers(
                    #     color_image.copy(), corners, ids)

                # # Show images
                # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Aruco Frames', color_image)
                cv2.waitKey(1)

    finally:
        # Stop streaming
        pipeline.stop()

    with open(f'tags/block_{block_id}_info.pkl', 'wb') as f:
        pickle.dump(info, f)

if __name__ == '__main__':
    main()
