#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion

from panda_vision.srv import GetBlockPoses, GetBlockPosesResponse
from panda_vision.msg import NamedPose

# this would be hard coded by us, to match the blocks given in run_stack_panda.py
tag_block_association = {'tagID0':'block0', 'tagID1':'block1', 'tagID2':'block2'}

def handle_get_block_poses(req):
    # call vision system...
    # example of what is returned
    pose_0 = PoseStamped()
    pose_0.pose.position = Point(0.65, 0.3, 0)
    pose_0.pose.orientation = Quaternion(0, 0, 0, 1)
    
    pose_1 = PoseStamped()
    pose_1.pose.position = Point(0.65, 0.15, 0)
    pose_1.pose.orientation = Quaternion(0, 0, 0, 1)
    
    pose_2 = PoseStamped()
    pose_2.pose.position = Point(0.65, 0.0, 0)
    pose_2.pose.orientation = Quaternion(0, 0, 0, 1)
    
    poses = {'tagID0': pose_0, 
                'tagID1': pose_1,
                'tagID2': pose_2}
    
    #
    named_poses = []
    for tagID, pose in poses.items():
        block_name = tag_block_association[tagID]
        named_poses.append(NamedPose(block_name, pose))
    
    return GetBlockPosesResponse(named_poses)
    
def get_block_poses_server():
    rospy.init_node('get_block_poses_server')
    rospy.Service('get_block_poses', GetBlockPoses, handle_get_block_poses)
    print('Get block poses server ready.')
    rospy.spin()
    
if __name__ == '__main__':
    get_block_poses_server()