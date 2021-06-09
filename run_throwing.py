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


def create_objects():
    """ Initializes a set of test objects """
    b1 = ThrowingBall(bounciness = 0.85)

    b2 = ThrowingBall(color = [0,0,1], mass = 1.2, radius = 0.03,
                      air_drag_angular = 1e-5, friction_coef = 0.85,
                      rolling_resistance = 1e-4, bounciness = 0.4)

    b3 = ThrowingBall(color = [0,0.6,0], mass=0.7, radius=0.05,
                      friction_coef = 0.2, rolling_resistance = 5e-3,
                      bounciness = 0.2)

    objects = [b1, b2, b3]
    return objects


def main(objects, args):
    """ Creates an agent and run some simulations """
    num_objs = min(args.num_objects, len(objects))
    agent = ThrowingAgent(objects[:num_objs+1])

    for i in range(args.num_sims):
        print(f"\nRunning simulation {i}")
        act = agent.sample_action()
        agent.run(act, do_animate=args.animate, do_plot=args.plot)
        print(f"Simulation {i} done")


if __name__=="__main__":
    args = parse_args()
    objects = create_objects()
    main(objects, args)
