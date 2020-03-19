import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import random
import copy
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from actions import PushAction
from filter_utils import create_uniform_particles, ParticleDistribution
from block_utils import Environment, Object, Position, Dimensions, World, \
                        Pose, Quaternion, Color, get_com_ranges, Contact, \
                        get_poses_from_contacts

TRUE_OBS_COV = 0.00004*np.eye(3)    # covariance used when add noise to observations
OBS_MODEL_COV = 0.00004*np.eye(3)   # covariance used in observation model
T = 50                              # number of steps to simulate per contact state
I = 3                               # number of contact states to try
N = 100                             # number of particles
D = 3                               # dimensions of a single particle

def add_noise(pose):
    pos = Position(*multivariate_normal.rvs(mean=pose.pos, cov=TRUE_OBS_COV))
    orn = pose.orn
    return Pose(pos, orn)

def plot_particles(ax, particles, weights, t=None):
    for particle, weight in zip(particles, weights):
        ax.scatter(*particle, s=10, c=str(1-weight))

    X = particles[:,0]
    Y = particles[:,1]
    Z = particles[:,2]

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('t='+str(t))  # need to stop from scrolling down figure
    #ax.view_init(elev=100., azim=0.0)  # top down view of x-y plane
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.draw()
    plt.pause(0.1)

def filter_world(p_true_world, args):
    if args.plot:
        plt.ion()
        fig = plt.figure()
        ax = Axes3D(fig)
    
    for i in range(I):
        true_world = copy.deepcopy(p_true_world)

        # create particle worlds for obj_b's COM
        com_ranges = get_com_ranges(true_world.objects[1])
        com_particle_dist = create_uniform_particles(N, D, com_ranges)
        weights = com_particle_dist.weights
        particle_worlds = [copy.deepcopy(true_world) for particle in com_particle_dist.particles]
        for (com, particle_world) in zip(com_particle_dist.particles, particle_worlds):
            particle_world.objects[1].com = com

        env = Environment([true_world]+particle_worlds, vis_sim=args.vis)

        # action to apply to all worlds
        action = PushAction(block_pos=true_world.get_pose(true_world.objects[1]).pos,
                            direction=PushAction.get_random_dir(),
                            timesteps=T)

        for t in range(T):
            env.step(action=action)
        
            # get ground truth object_b pose (observation)
            objb_pose = true_world.get_pose(true_world.objects[1])
            objb_pose = add_noise(objb_pose)

            # update all particle weights
            new_weights = []

            for pi, (particle_world, old_weight) in enumerate(zip(particle_worlds, weights)):
                particle_objb_pose = particle_world.get_pose(particle_world.objects[1])
                obs_model = multivariate_normal.pdf(objb_pose.pos,
                                                    mean=particle_objb_pose.pos,
                                                    cov=OBS_MODEL_COV)
                new_weight = old_weight * obs_model
                new_weights.append(new_weight)

            # normalize particle weights
            weights_sum = sum(new_weights)
            weights = np.divide(new_weights, weights_sum)
            # print('max particle weight: ', max(weights))

            if args.plot and not t % 5:
                # visualize particles (it's very slow)
                plot_particles(ax, com_particle_dist.particles, weights, t=t)

        env.disconnect()
        env.cleanup()

    return ParticleDistribution(com_particle_dist.particles, weights)
