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


    def sample_action(self):
        """
        Samples an action, which consists of a ball and release velocities
        """

        # Sample a random ball
        b = np.random.choice(self.objects)

        # Sample random velocities
        v = 5
        ang = np.random.uniform(np.pi/8, 3*np.pi/8)
        w = np.random.uniform(-10, 10)

        # Package up the action
        init_vel = [
            v * np.cos(ang),
            v * np.sin(ang),
            w, 
        ]
        return ThrowingAction(b, init_pos=[0,0,0], init_vel=init_vel)


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
        return True
  