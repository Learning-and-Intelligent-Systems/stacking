""" Massachusetts Institute of Technology

Izzybrand, 2020
"""

import rospy
from std_msgs.msg import String

from block_pose_est impot ArucoBlockVision


def main():
    # setup the ros node
    pub = rospy.Publisher('block_pose', String, queue_size=10)
    rospy.init_node("block_pose_pub")

    def publish_callback(block_id, X_CO):
        pub.publish(f'{block_id} at {X_CO}')

    ArucoBlockVision(publish_callback)


if __name__ == '__main__':
    main()
