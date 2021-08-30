"""
Test script for ball throwing domain

Copyright 2021 Massachusetts Institute of Technology
"""

import argparse
import numpy as np
from matplotlib import pyplot as plt

from agents.throwing_agent import ThrowingAgent
from learning.domains.throwing.entities import ThrowingBall, ThrowingAction


def parse_args():
    """ Parses command-line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-objects', type=int, default=3)
    args = parser.parse_args()
    return args

def create_objects(num_objects):
    masses = np.linspace(0.5, 1.5, num_objects)
    r = 0.025
    return [ThrowingBall(mass=m, radius=r) for m in masses]

def main(objects, args):
    """ Creates an agent and run some simulations """
    agent = ThrowingAgent(objects)
    ang = 3*np.pi/8
    w = 0

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, aspect="equal")

    for o in objects:
        act = ThrowingAction.from_vector(o, [ang, w])
        results = agent.run(act, return_full_results=True)
        agent.simulator.plot_results(o, results["state"], label=f"Mass: {o.mass}kg", ax=ax)

    plt.legend()
    plt.show()

if __name__=="__main__":
    args = parse_args()
    objects = create_objects(args.num_objects)
    main(objects, args)
