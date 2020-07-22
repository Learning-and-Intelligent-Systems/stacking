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
    blocks = get_adversarial_blocks(num_blocks=args.num_blocks)

    if args.agent == 'teleport':
        agent = TeleportAgent(blocks, NOISE)
    elif args.agent == 'panda':
        agent = PandaAgent(blocks, NOISE, teleport=True)
    else:
        raise NotImplementedError()

    # construct a world containing those blocks
    beliefs = [ParticleBelief(block, 
                              N=200, 
                              plot=True, 
                              vis_sim=args.vis,
                              noise=NOISE) for block in blocks]
    # agent._add_text('Ready?')
    input('Start?')
    for b_ix, (block, belief) in enumerate(zip(blocks, beliefs)):
        print('Running filter for', block.name)
        for interaction_num in range(5):
            print("Interaction number: ", interaction_num)
            action = plan_action(belief, exp_type='reduce_var', action_type='place')
            observation = agent.simulate_action(action, b_ix, T=50)
            # agent._add_text('Updating particle belief')
            belief.update(observation)
            block.com_filter = belief.particles

        print(belief.estimated_coms[-1], block.com)

        

    # find the tallest tower
    print('Finding tallest tower.')
    # agent._add_text('Planning tallest tower')
    tp = TowerPlanner()
    tallest_tower = tp.plan(blocks)

    # and visualize the result
    agent.simulate_tower(tallest_tower, vis=True, T=2500, save_tower=args.save_tower)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num-blocks', type=int, default=3)
    parser.add_argument('--save-tower', action='store_true')
    parser.add_argument('--agent', choices=['teleport', 'panda'], default='teleport')
    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    # test_exploration(args)

    main(args)
