"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import random
from copy import deepcopy
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from actions import *
from filter_utils import *
from block_utils import *

class ParticleBelief:
    def __init__(self, block, N=200, plot=False, vis_sim=False):
        self.block = deepcopy(block)
        self.plot = plot                        # plot the particles
        self.vis_sim = vis_sim                  # display the pybullet simulator

        self.TRUE_OBS_COV = 0.0015*np.eye(3)    # covariance used when add noise to observations
        self.OBS_MODEL_COV = 0.0015*np.eye(3)   # covariance used in observation model
        self.N = N                              # number of particles
        self.D = 3                              # dimensions of a single particle

        self.setup()

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
            ax.scatter(*particle, s=10, color=(weight,0,1-weight))

        ax.set_title('t='+str(t))  # need to stop from scrolling down figure
        #ax.view_init(elev=100., azim=0.0)  # top down view of x-y plane

        plt.draw()
        plt.pause(0.1)

    def update(self, observation):
        # observation is a tuple (action, rot, timesteps, pose)
        if self.plot:
            plt.ion()
            fig = plt.figure()
            ax = Axes3D(fig)

        if self.plot: self.setup_ax(ax, true_world.objects[1])

        # resample the distribution
        self.particles = sample_and_wiggle(self.particles, self.experience,
            self.OBS_MODEL_COV, self.block, self.com_ranges)

        self.experience.append(observation)
        action, rot, T, end_pose = observation

        particle_blocks = [deepcopy(self.block) for particle in self.particles.particles]
        for (com, particle_block) in zip(self.particles.particles, particle_blocks):
            particle_block.com = com
        particle_worlds = [make_platform_world(pb, rot) for pb in particle_blocks]
        env = Environment(particle_worlds, vis_sim=self.vis_sim)
        for t in range(T):
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

        if self.plot and not t % 5:
            # visualize particles (it's very slow)
            self.plot_particles(ax, self.particles.particles, weights, t=t)

        mean = np.array(self.particles.particles).T@np.array(self.particles.weights)
        # print('Mean COM', mean, np.diag(np.cov(self.particles.particles, rowvar=False, aweights=self.particles.weights+1e-3)))

        # if self.block.com is not None:
        #     print('True COM', self.block.com)
        #     print('Error COM', np.linalg.norm(self.block.com-mean))

        self.estimated_coms.append(mean)

        env.disconnect()
        env.cleanup()

        return self.particles, self.estimated_coms
