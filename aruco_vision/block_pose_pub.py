""" Massachusetts Institute of Technology

Izzybrand, 2020
"""

import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion

from aruco_vision.msg import NamedPose
from block_pose_est import BlockPoseEst
from rotation_util import *


def main():
    # setup the ros node
    pub = rospy.Publisher('block_pose', NamedPose, queue_size=10)
    rospy.init_node("block_pose_pub")

    def publish_callback(block_id, X_CO):
        # create a pose message
        p = PoseStamped()
        p.header.stamp = rospy.Time.now()

        # populate with the pose information
        R_CO, t_CO = pose_matrix_to_Rt(X_CO)
        p.pose.position = Point(*t_CO)
        p.pose.orientation = Quaternion(*rot_to_quat(R_CO))

        # and publish the named pose
        pub.publish(NamedPose(str(block_id), p))

    bpe = BlockPoseEst(publish_callback)

    while not rospy.is_shutdown():
        bpe.step()

    bpe.close()
    


if __name__ == '__main__':
    main()
