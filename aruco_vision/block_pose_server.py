#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point, Quaternion

from panda_vision.srv import GetBlockPoses, GetBlockPosesResponse
from panda_vision.msg import NamedPose

# NOTE(izzy): in order to be able to continuously collect frames from the
# camera and also handle server requests, I've decided to have a publisher
# node which constantly publishes block poses as they are observed, and
# the block_pose_server is a subscriber to that node

poses = {}

def block_pose_callback(data):
    poses[block_id] = pose

def handle_get_block_poses(req):
    for block_id, pose in poses.items():
        block_id
        named_poses.append(NamedPose(block_id, pose))
    
    return GetBlockPosesResponse(named_poses)
    
def block_pose_server():
    rospy.init_node('block_pose_server')
    rospy.Service('get_block_poses', GetBlockPoses, handle_get_block_poses)
    print('Get block poses server ready.')
    rospy.spin()
    
if __name__ == '__main__':
    block_pose_server()