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
    NOISE=0.00005

    # get a bunch of random blocks
    blocks = get_adversarial_blocks()

    if args.agent == 'teleport':
        agent = TeleportAgent(blocks, NOISE)
    else:
        raise NotImplementedError()

    # construct a world containing those blocks
    for b_ix, block in enumerate(blocks):
        print('Running filter for', block.name)
        belief = ParticleBelief(block, 
                                N=200, 
                                plot=args.plot, 
                                vis_sim=args.vis,
                                noise=NOISE)
        for interaction_num in range(10):
            print("Interaction number: ", interaction_num)
            action = plan_action(belief, exp_type='random', action_type='place')
            observation = agent.simulate_action(action, b_ix)
            belief.update(observation)
            block.com_filter = belief.particles
        print(belief.estimated_coms[-1], block.com)

    # find the tallest tower
    print('Finding tallest tower.')
    tp = TowerPlanner()
    tallest_tower = tp.plan(blocks)

    # and visualize the result
    agent.simulate_tower(tallest_tower, vis=True, T=250, save_tower=args.save_tower)


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
