#!/usr/bin/env python

import time

import rospy
from aruco_vision.srv import GetBlockPoses


if __name__ == "__main__":
    while True:
        print("Requesting")
        rospy.wait_for_service('get_block_poses')
        try:
            get_block_poses = rospy.ServiceProxy('get_block_poses', GetBlockPoses)
            print(get_block_poses())
            
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        time.sleep(1)
