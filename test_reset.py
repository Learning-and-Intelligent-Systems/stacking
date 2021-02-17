import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pybullet as p

from actions import plan_action
from agents.teleport_agent import TeleportAgent
from agents.panda_agent import PandaAgent
from block_utils import Object, Dimensions, Position, Color, get_adversarial_blocks
from learning.domains.towers.generate_tower_training_data import sample_random_tower, build_tower
from particle_belief import ParticleBelief
from tower_planner import TowerPlanner
import pb_robot
from tamp.misc import load_blocks

def main(args):
    NOISE=0.00005

    # get a bunch of random blocks
    # if args.use_vision:
    if True:
        blocks = load_blocks(args.blocks_file, 10, [1])
    else:
        blocks = get_adversarial_blocks(num_blocks=args.num_blocks)

    input('Please make sure blocks are behind the robot.')
    agent = PandaAgent(blocks, NOISE,
        use_platform=False, teleport=False,
        use_action_server=args.use_action_server,
        use_vision=args.use_vision, real=args.real)
    original_poses = [b.get_base_link_pose() for b in agent.pddl_blocks]

    while True:
        input('Set blocks to clean up position. Ready?')
        agent._update_block_poses()
        
        # Check which blocks are in the front of the table.
        agent.moved_blocks = set()
        for b in agent.pddl_blocks:
            pos = b.get_base_link_point()
            if pos[0] > 0.05:
                agent.moved_blocks.add(b)

        # Clean up.
        agent.plan_reset_parallel(original_poses, real=args.real, T=2500, import_ros=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num-blocks', type=int, default=4)
    parser.add_argument('--num-towers', type=int, default=100)
    parser.add_argument('--save-tower', action='store_true')
    parser.add_argument('--use-action-server', action='store_true')
    parser.add_argument('--use-vision', action='store_true', help='get block poses from AR tags')
    parser.add_argument('--blocks-file', type=str, default='learning/domains/towers/final_block_set.pkl')
    parser.add_argument('--real', action='store_true', help='run on real robot')
    parser.add_argument('--show-frames', action='store_true')
    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    # test_exploration(args)

    main(args)
