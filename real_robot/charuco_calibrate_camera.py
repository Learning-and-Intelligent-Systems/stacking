""" Massachusetts Institute of Technology

Izzybrand, 2020
""" 
import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco


square_len_cm = 25.5/7.
marker_len_cm = square_len_cm * 0.8
# create the aruco board
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(7, 5, square_len_cm, marker_len_cm, aruco_dict)
imboard = board.draw((2000,3000))
cv2.imwrite('charuco_board.png', imboard)

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
    allCorners = []
    allIds = []
    for i in range(1000):
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
        res = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(res[0])>0:
            res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and i%5==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

            cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    imsize = gray.shape

    #Calibration fails for lots of reasons.
    cal = cv2.aruco.calibrateCameraCharuco(allCorners, allIds, board, imsize, None, None)
    print('mtx:', cal[1])
    print('dist:', cal[2])

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()