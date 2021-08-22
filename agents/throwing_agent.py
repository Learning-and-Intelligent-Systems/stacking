"""
Agent for learning in the 2D throwing domain

Copyright 2021 Massachusetts Institute of Technology
"""

import numpy as np
from learning.domains.throwing.entities import ThrowingBall, ThrowingAction
from learning.domains.throwing.simulator import ThrowingSimulator


class ThrowingAgent:

    def __init__(self, objects=[]):

        self.objects = objects
        self.simulator = ThrowingSimulator(self.objects, 
                                           dt=0.0005,
                                           tmax=5)


    def sample_action(self, b=None):
        """
        Samples an action, which consists of a ball and release velocities
        """

        # Sample a random ball
        if b is None: b = np.random.choice(self.objects)
        # and a random throw
        vec = ThrowingAction.random_vector()
        # create the ThrowingAction
        return ThrowingAction.from_vector(b, vec)


    def run(self, action, do_animate=False, do_plot=False):
        """ 
        Simulates a throwing action and collects results for learning
        """
        results = self.simulator.simulate(action, 
                                          do_animate=do_animate,
                                          do_plot=do_plot)

        # TODO: Parse through the simulation results and pick desired criteria
        # (e.g. max x direction, number of bounces, etc.)
        # Then, return learning signal
        return results["state"][0,-1]
  