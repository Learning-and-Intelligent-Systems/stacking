import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

from actions import plan_action
from agents.teleport_agent import TeleportAgent
from block_utils import Object, Dimensions, Position, Color, get_adversarial_blocks
from particle_belief import ParticleBelief
from tower_planner import TowerPlanner


def main(args):
    # get a bunch of random blocks
    blocks = get_adversarial_blocks()

    if args.agent == 'teleport':
        agent = TeleportAgent()
    else:
        raise NotImplementedError()

    # construct a world containing those blocks
    for block in blocks:
        # new code
        print('Running filter for', block.name)
        belief = ParticleBelief(block, N=100, plot=args.plot, vis_sim=args.vis)
        for interaction_num in range(5):
            print("Interaction number: ", interaction_num)
            action = plan_action(belief, exp_type='random', action_type='place')
            observation = agent.simulate_action(action, block)
            belief.update(observation)
            block.com_filter = belief.particles

    # find the tallest tower
    print('Finding tallest tower.')
    tp = TowerPlanner()
    tallest_tower = tp.plan(blocks)

    # and visualize the result
    agent.simulate_tower(tallest_tower, vis=True, T=100, save_tower=args.save_tower)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--save-tower', action='store_true')
    parser.add_argument('--agent', choices=['teleport', 'panda'], default='teleport')
    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    # test_exploration(args)

    main(args)
