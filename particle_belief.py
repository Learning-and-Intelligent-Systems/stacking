"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil

from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

from actions import make_platform_world, plan_action
from agents.panda_agent import PandaAgent
from agents.teleport_agent import TeleportAgent
from base_class import BeliefBase
from block_utils import get_adversarial_blocks, get_com_ranges, \
                        Environment, ParticleDistribution
from filter_utils import create_uniform_particles, sample_and_wiggle


class ParticleBelief(BeliefBase):
    def __init__(self, block, noise, N=200, plot=False, vis_sim=False):
        self.block = deepcopy(block)
        self.plot = plot                        # plot the particles
        self.vis_sim = vis_sim                  # display the pybullet simulator

        self.TRUE_OBS_COV = noise*np.eye(3)     # covariance used when add noise to observations
        self.OBS_MODEL_COV = noise*np.eye(3)    # covariance used in observation model
        self.N = N                              # number of particles
        self.D = 3                              # dimensions of a single particle

        self.setup()
        if self.plot:
            plt.ion()
            fig = plt.figure()
            self.ax = Axes3D(fig)

    def setup(self):
        self.com_ranges = get_com_ranges(self.block)
        self.particles = create_uniform_particles(self.N, self.D, self.com_ranges)
        self.experience = []
        self.estimated_coms = []

    def setup_ax(self, ax, obj):
        ax.clear()
        halfdim = max(obj.dimensions)
        ax.set_xlim(-halfdim, halfdim)
        ax.set_ylim(-halfdim, halfdim)
        ax.set_zlim(-halfdim, halfdim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    def plot_particles(self, ax, particles, weights, t=None, true_com=None):
        for particle, weight in zip(particles, weights):
            alpha = 0.25 + 0.75 * weight
            ax.scatter(*particle, s=10, color=(0, 0, 1), alpha=alpha)

        ax.scatter(*self.block.com, s=10, color=(1,0,0))
        ax.set_title('Particle Dist')  # need to stop from scrolling down figure
        #ax.view_init(elev=100., azim=0.0)  # top down view of x-y plane

        plt.draw()
        plt.pause(0.1)

    def update(self, observation):
        # observation is a tuple (action, rot, timesteps, pose)
        
        # resample the distribution
        self.particles = sample_and_wiggle(self.particles, self.experience,
            self.OBS_MODEL_COV, self.block, self.com_ranges)

        self.experience.append(observation)
        action, T, end_pose = observation

        particle_blocks = [deepcopy(self.block) for particle in self.particles.particles]
        for (com, particle_block) in zip(self.particles.particles, particle_blocks):
            particle_block.com = com
        particle_worlds = [make_platform_world(pb, action) for pb in particle_blocks]
        env = Environment(particle_worlds, vis_sim=self.vis_sim)
        for _ in range(T):
            env.step(action=action)

        # update all particle weights
        new_weights = []

        for pi, (particle_world, old_weight) in enumerate(zip(particle_worlds, self.particles.weights)):
            particle_end_pose = particle_world.get_pose(particle_world.objects[1])            
            obs_model = multivariate_normal.pdf(end_pose.pos,
                                                mean=particle_end_pose.pos,
                                                cov=self.OBS_MODEL_COV)
            new_weight = old_weight * obs_model
            new_weights.append(new_weight)

        # normalize particle weights
        new_weights = np.array(new_weights)/np.sum(new_weights)
        # and update the particle distribution with the new weights
        self.particles = ParticleDistribution(self.particles.particles, new_weights)

        if self.plot:
            # visualize particles (it's very slow)
            #plt.ion()
            #fig = plt.figure()
            #self.ax = Axes3D(fig)
            self.setup_ax(self.ax, self.block)
            self.plot_particles(self.ax, self.particles.particles, new_weights)

        mean = np.array(self.particles.particles).T@np.array(self.particles.weights)
        # print('Mean COM', mean, np.diag(np.cov(self.particles.particles, rowvar=False, aweights=self.particles.weights+1e-3)))

        # if self.block.com is not None:
        #     print('True COM', self.block.com)
        #     print('Error COM', np.linalg.norm(self.block.com-mean))

        self.estimated_coms.append(mean)

        env.disconnect()
        env.cleanup()

        return self.particles, self.estimated_coms


# =============================================================
"""
A script that tests the particle filter by reporting the error on
the CoM estimate as we get more observations.
"""
def plot_com_error(errors_random, errors_var):

    for tx in range(0, len(errors_var[0][0])):
        err_rand, err_var = 0, 0
        for bx in range(0, len(errors_var)):
            true = np.array(errors_var[bx][1])
            guess_rand = errors_random[bx][0][tx]
            guess_var = errors_var[bx][0][tx]
            err_var += np.linalg.norm(true-guess_var)
            err_rand += np.linalg.norm(true-guess_rand)
        plt.scatter(tx, err_rand/len(errors_var), c='r')
        plt.scatter(tx, err_var/len(errors_var), c='b')
    plt.show()    

"""
Notes on tuning the particle filter.
- When plot=True for ParticleBelief, we want to see the particles (blue) become 
  more tightly distributed around the true CoM (red). 
- Make sure some particles are initialized near the true CoM.
- Check resampling-step if particles don't converge to true CoM.
-- Are M-H steps being accepted? Are we removing unlikely samples?
- I have observed that for PlaceAction, n_particles=200, and n_actions=10, 
  the distribution converges pretty tightly for all adversarial blocks.
- If the particles jump around too much, the true noise might be too large.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['teleport', 'panda'], default='teleport')
    parser.add_argument('--n-particles', type=int, default=10)
    parser.add_argument('--n-actions', type=int, default=2)
    args = parser.parse_args()
    NOISE=0.00005

    # get a bunch of random blocks
    blocks = get_adversarial_blocks()

    if args.agent == 'teleport':
        agent = TeleportAgent(blocks, NOISE)
    else:
        agent = PandaAgent(blocks, NOISE)
    
    
    # construct a world containing those blocks
    for b_ix, block in enumerate(blocks):
        # new code
        print('Running filter for', block.name, block.dimensions)
        belief = ParticleBelief(block, 
                                N=args.n_particles, 
                                plot=True, 
                                vis_sim=False,
                                noise=NOISE)
        for interaction_num in range(args.n_actions):
            print('----------')
            # print(belief.particles.particles[::4, :])
            print("Interaction number: ", interaction_num)
            action = plan_action(belief, exp_type='random', action_type='place')
            observation = agent.simulate_action(action, b_ix, vis_sim=False)
            belief.update(observation)
            block.com_filter = belief.particles  

            est = belief.estimated_coms[-1]
            true = np.array(block.com)
            error = np.linalg.norm(est-true)
            print('Estimated CoM:', est)
            print('True:', true)  
            print('Error:', error)

        

