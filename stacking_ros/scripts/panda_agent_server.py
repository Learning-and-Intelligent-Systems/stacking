#!/usr/bin/env python3.7
"""
Runs a PandaAgent as a server for multi-machine active learning
"""

import rospy
import argparse
from agents.panda_agent import PandaAgent

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-blocks', type=int, default=4)
    parser.add_argument('--real', default=False, action='store_true')
    parser.add_argument('--use-vision', default=False, action='store_true')
    parser.add_argument('--blocks-file', default='', type=str)
    args = parser.parse_args()

    # Load the blocks
    if len(args.blocks_file) > 0:
        import pickle
        with open(args.blocks_file, 'rb') as handle:
            blocks = pickle.load(handle)
        block_init_xy_poses = None
    else:
        from block_utils import get_adversarial_blocks
        blocks = get_adversarial_blocks(num_blocks=args.num_blocks)

    # Create Panda agent and leave it running as ROS node
    agent = PandaAgent(blocks, 
                       use_vision=args.use_vision, 
                       real=args.real,
                       use_action_server=True, 
                       use_learning_server=True)
    rospy.spin()
