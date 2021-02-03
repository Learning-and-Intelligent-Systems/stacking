import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle

from actions import plan_action
from agents.panda_agent import PandaAgent
from block_utils import Object, Dimensions, Pose, Position, Quaternion, Color, get_rotated_block
from tower_planner import TowerPlanner

def main(args):
    NOISE=0.00005

    # define real world block params and initial poses
    #TODO: Load blocks from pickle file.
    if args.use_vision:
        with open(args.blocks_file, 'rb') as handle:
            blocks = pickle.load(handle)
        block_init_xy_poses = None  # These will be initialized randomly but updated by the vision system.
    else:
        block0 = Object('block0', Dimensions(.0381,.0318,.0635), 1.0, Position(0,0,0), Color(1,0,0))
        block1 = Object('block1', Dimensions(.0381,.0587,.0635), 1.0, Position(0,0,0), Color(0,0,1))
        block2 = Object('block2', Dimensions(.0635,.0381,.0746), 1.0, Position(0,0,0), Color(0,1,0))
        blocks = [block0, block1, block2]

        block_init_xy_poses = [Pose(Position(0.65,0.3,0), Quaternion(0,0,0,1)),
                               Pose(Position(0.65,0.15,0), Quaternion(0,0,0,1)),
                               Pose(Position(0.65,0.0,0), Quaternion(0,0,0,1))]

    panda = PandaAgent(blocks, NOISE,
                       use_platform=False,
                       block_init_xy_poses=block_init_xy_poses,
                       teleport=False,
                       use_vision=args.use_vision)

    # for now hard-code a tower, but in the future will be supplied from
    # active data collection or tower found through planning for evaluation
    tower_blocks = copy.copy(blocks)
    if args.use_vision:
        tower_poses = [Pose(Position(0.5,-0.25,0.0725), Quaternion(0,0,0,1)),
                        Pose(Position(0.5,-0.25,0.18), Quaternion(0,0,0,1)),
                        Pose(Position(0.5,-0.25,0.28), Quaternion(0,0,0,1))]
    else:
        tower_poses = [Pose(Position(0.3,0.25,.0318), Quaternion(0,0,0,1)),
                        Pose(Position(0.3,0.25,.0953), Quaternion(0,0,0,1)),
                        Pose(Position(0.3,0.25,.1643), Quaternion(0,0,0,1))]

    tower = []
    for block, pose in zip(tower_blocks, tower_poses):
        block.set_pose(pose)
        block = get_rotated_block(block) # NOTE: have to do to set rotations field of block
        tower.append(block)

    # and execute the resulting plan.
    panda.simulate_tower(tower, base_xy=(0.5, -0.25), real=args.real, vis=True, T=2500)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true', help='run on real robot')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use-vision', action='store_true', help='get block poses from AR tags')
    parser.add_argument('--blocks-file', type=str, default='learning/domains/towers/initial_block_set.pkl')
    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    if args.real:
        import rospy
        rospy.init_node('path_execution')

    main(args)
