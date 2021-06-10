"""
Test script for ball throwing domain

Copyright 2021 Massachusetts Institute of Technology
"""

import pdb
import argparse
from agents.throwing_agent import ThrowingAgent
from learning.domains.throwing.entities import ThrowingBall


def parse_args():
    """ Parses command-line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--num-objects', type=int, default=4)
    parser.add_argument('--num-sims', type=int, default=10)
    args = parser.parse_args()
    if args.debug: 
        pdb.set_trace()
    return args

def create_random_objects(num_objects):
    return [ThrowingBall.random() for _ in range(num_objects)]

def main(objects, args):
    """ Creates an agent and run some simulations """
    agent = ThrowingAgent(objects)

    for i in range(args.num_sims):
        print(f"\nRunning simulation {i}")
        act = agent.sample_action()
        agent.run(act, do_animate=args.animate, do_plot=args.plot)
        print(f"Simulation {i} done")


if __name__=="__main__":
    args = parse_args()
    objects = create_random_objects(args.num_objects)
    main(objects, args)
