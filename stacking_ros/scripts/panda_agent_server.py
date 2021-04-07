#!/usr/bin/env python3
"""
Runs a PandaAgent as a server for multi-machine active learning
"""

import rospy
import argparse
from agents.panda_agent import PandaAgent
from tamp.misc import load_blocks

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-blocks', type=int, default=4)
    parser.add_argument('--real', default=False, action='store_true')
    parser.add_argument('--use-vision', default=False, action='store_true')
    parser.add_argument('--blocks-file', default='learning/domains/towers/final_block_set_10.pkl', type=str)
    args = parser.parse_args()

    # Load the blocks
    blocks = load_blocks(fname=args.blocks_file,
                         num_blocks=args.num_blocks)
    block_init_xy_poses = None

    # Create Panda agent and leave it running as ROS node
    agent = PandaAgent(blocks,
                       use_vision=args.use_vision,
                       real=args.real,
                       use_action_server=True,
                       use_learning_server=True)
    print("Panda agent server ready!")
    rospy.spin()
