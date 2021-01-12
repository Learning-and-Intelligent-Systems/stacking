import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

from actions import plan_action
from agents.teleport_agent import TeleportAgent
from agents.panda_agent import PandaAgent
from block_utils import Object, Dimensions, Position, Color, get_adversarial_blocks
from learning.domains.towers.generate_tower_training_data import sample_random_tower
from particle_belief import ParticleBelief
from tower_planner import TowerPlanner


def main(args):
    NOISE=0.00005

    # get a bunch of random blocks
    blocks = get_adversarial_blocks(num_blocks=args.num_blocks)
    agent = PandaAgent(blocks, NOISE, use_platform=False, teleport=False)

    for tx in range(0, 10):
        # Build a random tower out of 5 blocks.
        n_blocks = np.random.randint(2, args.num_blocks + 1)
        tower_blocks = np.random.choice(blocks, n_blocks, replace=False)

        tower = sample_random_tower(tower_blocks)

        # and execute the resulting plan.
        agent.simulate_tower(tower,
                             base_xy=(0.5, -0.3), 
                             vis=True, 
                             T=1000, 
                             save_tower=args.save_tower)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num-blocks', type=int, default=4)
    parser.add_argument('--save-tower', action='store_true')
    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    # test_exploration(args)

    main(args)
