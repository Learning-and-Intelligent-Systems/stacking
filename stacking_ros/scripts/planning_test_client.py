#!/usr/bin/env python3.7
"""
Test Client for ROS Planning Service
"""

import rospy
import actionlib
import pb_robot
from stacking_ros.msg import BodyInfo, TaskAction, TaskPlanGoal
from block_utils import Object, Dimensions, Position, Color, get_adversarial_blocks
from tamp.misc import setup_panda_world


def body_to_ros(body):
    """
    Convert Body objects to ROS message
    """
    msg = BodyInfo()
    msg.name = body.get_body_name()
    msg.mass = body.get_mass()
    msg.com.x, msg.com.y, msg.com.z = body.get_point()
    msg.dimensions.x, msg.dimensions.y, msg.dimensions.z = body.get_dimensions()
    pose = body.get_base_link_pose()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[0]
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = pose[1]
    return msg


class PlanningClient():
    def __init__(self):
        # Arguments
        num_blocks = 4
        use_platform = False
        block_init_xy_poses = None

        # Create ROS Action Client
        rospy.init_node("test_planning_client")

        self._planning_client_id = pb_robot.utils.connect(use_gui=False)
        self.robot = pb_robot.panda.Panda()
        blocks = get_adversarial_blocks(num_blocks=num_blocks)
        self.pddl_blocks, self.platform_table, self.platform_leg, self.table, self.frame, wall = \
                setup_panda_world(self.robot, blocks, block_init_xy_poses, use_platform=use_platform) 

        for block in self.pddl_blocks:
            msg = body_to_ros(block)
            print(msg)


if __name__=="__main__":
    c = PlanningClient()
