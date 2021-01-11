import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

from actions import plan_action
from agents.teleport_agent import TeleportAgent
from agents.panda_agent import PandaAgent
from block_utils import Object, Dimensions, Position, Color, get_adversarial_blocks
from particle_belief import ParticleBelief
from tower_planner import TowerPlanner


def main(args):
    NOISE=0.00005

    # get a bunch of random blocks
    blocks = get_adversarial_blocks(num_blocks=4)
    agent = PandaAgent(blocks, NOISE, use_platform=False, teleport=False)

    input()
    for tx in range(0, 10):
        # TODO: Build a random tower out of 5 blocks.

        # and execute the resulting plan.
        agent.simulate_tower(tallest_tower, vis=True, T=2500, save_tower=args.save_tower)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num-blocks', type=int, default=3)
    parser.add_argument('--save-tower', action='store_true')
    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    # test_exploration(args)

    main(args)
