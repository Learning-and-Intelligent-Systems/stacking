from actions import make_platform_world
from block_utils import Environment, World, add_noise

from copy import copy

import numpy as np


class TeleportAgent:
    def __init__(self, noise):
        self.noise = noise

    def simulate_action(self, action, real_block, T=50, vis_sim=False):
        # set up the environment with the real block
        true_world = make_platform_world(real_block, action)
        env = Environment([true_world], vis_sim=vis_sim)
        # configure the duration of the action
        action.timesteps = T
        # run the simulator
        for _ in range(T):
            env.step(action=action)
        # get ground truth object_b pose (observation)
        end_pose = true_world.get_pose(true_world.objects[1])
        end_pose = add_noise(end_pose, self.noise*np.eye(3))
        observation = (action, T, end_pose)
        # turn off the sim
        env.disconnect()
        env.cleanup()
        # and return the observed trajectory
        return observation

    def simulate_tower(self, tower, vis=True, T=60, copy_blocks=True, save_tower=False):
        if copy_blocks:
            tower = [copy(block) for block in tower]
        world = World(tower)

        env = Environment([world], 
                          vis_sim=vis, 
                          use_hand=False, 
                          save_tower=save_tower)
        for _ in range(T):
            env.step(vis_frames=vis)
        env.disconnect()
        env.cleanup()

        return world.get_poses()


