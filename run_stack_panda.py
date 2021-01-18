import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy

from actions import plan_action
from agents.panda_agent import PandaAgent
from block_utils import Object, Dimensions, Pose, Position, Quaternion, Color, get_rotated_block
from tower_planner import TowerPlanner

def main(args):
    NOISE=0.00005
        
    # define real world block params and initial poses
    block0 = Object('block0', Dimensions(.04,.04,.1), 1.0, Position(0,0,0), Color(0,0,1))
    block1 = Object('block1', Dimensions(.04,.04,.1), 1.0, Position(0,0,0), Color(0,1,0))
    block2 = Object('block2', Dimensions(.04,.04,.1), 1.0, Position(0,0,0), Color(1,0,0))
    blocks = [block0, block1, block2]
    
    block_init_xy_poses = [Pose(Position(0.65,0.3,0), Quaternion(0,0,0,1)),
                        Pose(Position(0.65,0.15,0), Quaternion(0,0,0,1)),
                        Pose(Position(0.65,0.0,0), Quaternion(0,0,0,1))]
    panda = PandaAgent(blocks, NOISE, False, block_init_xy_poses=block_init_xy_poses, teleport=False)

    # for now hard-code a tower, but in the future will be supplied from
    # active data collection or tower found through planning for evaluation
    tower_blocks = copy.copy(blocks)
    tower_poses = [Pose(Position(0.3,0.25,.05), Quaternion(0,0,0,1)),
                    Pose(Position(0.3,0.25,.15), Quaternion(0,0,0,1)),
                    Pose(Position(0.3,0.25,.25), Quaternion(0,0,0,1))]
    
    tower = []
    for block, pose in zip(tower_blocks, tower_poses):
        block.set_pose(pose)
        block = get_rotated_block(block) # NOTE: have to do to set rotations field of block
        tower.append(block)
        
    # and execute the resulting plan.
    panda.simulate_tower(tower, real=args.real, vis=True, T=2500)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true', help='run on real robot')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    if args.real:
        import rospy
        rospy.init_node('path_execution')

    main(args)
