import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import random
import copy
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from actions import PushAction, make_platform_world, plan_action
from filter_utils import create_uniform_particles, ParticleDistribution, sample_and_wiggle
from block_utils import Environment, Object, Position, Dimensions, World, \
                        Pose, Quaternion, Color, get_com_ranges, Contact, \
                        rotation_group

TRUE_OBS_COV = 0.0015*np.eye(3)    # covariance used when add noise to observations
OBS_MODEL_COV = 0.0015*np.eye(3)   # covariance used in observation model
T = 50                              # number of steps to simulate per contact state
I = 10                               # number of contact states to try
N = 200                             # number of particles
D = 3                               # dimensions of a single particle

def add_noise(pose):
    pos = Position(*multivariate_normal.rvs(mean=pose.pos, cov=TRUE_OBS_COV))
    orn = pose.orn
    return Pose(pos, orn)

def setup_ax(ax, obj):
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

def plot_particles(ax, particles, weights, t=None, true_com=None):
    for particle, weight in zip(particles, weights):
        ax.scatter(*particle, s=10, color=(weight,0,1-weight))

    ax.set_title('t='+str(t))  # need to stop from scrolling down figure
    #ax.view_init(elev=100., azim=0.0)  # top down view of x-y plane

    plt.draw()
    plt.pause(0.1)

def filter_block(p_true_block, exp_type, args):
    if args.plot:
        plt.ion()
        fig = plt.figure()
        ax = Axes3D(fig)

    com_particle_dist = None
    experience = []

    estimated_coms = []
    for i in range(I):
        print('Explore action %d/%d' % (i, I))
        true_block = copy.deepcopy(p_true_block)
        if args.plot: setup_ax(ax, true_world.objects[1])

        # create particle worlds for obj_b's COM
        com_ranges = get_com_ranges(true_block)
        if com_particle_dist is None:
            com_particle_dist = create_uniform_particles(N, D, com_ranges)
        else:
            # update the distribution with the new weights
            print(weights)
            com_particle_dist = ParticleDistribution(com_particle_dist.particles, weights)
            # and resample the distribution
            com_particle_dist = sample_and_wiggle(com_particle_dist, experience, OBS_MODEL_COV, true_block, com_ranges)

        weights = com_particle_dist.weights
        particle_blocks = [copy.deepcopy(true_block) for particle in com_particle_dist.particles]
        for (com, particle_block) in zip(com_particle_dist.particles, particle_blocks):
            particle_block.com = com

        # Choose action to maximize variance of particles.
        if exp_type == 'reduce_var':
            rot, direc = plan_action(particle_blocks)
        elif exp_type == 'random':
            rot = random.choice(list(rotation_group()))
            direc = PushAction.get_random_dir()

        true_world = make_platform_world(true_block, rot)
        particle_worlds = [make_platform_world(pb, rot) for pb in particle_blocks]

        env = Environment([true_world]+particle_worlds, vis_sim=args.vis)

        # action to apply to all worlds
        action = PushAction(block_pos=true_world.get_pose(true_world.objects[1]).pos,
                            direction=direc,
                            timesteps=T)

        for t in range(T):
            env.step(action=action)

        # get ground truth object_b pose (observation)
        objb_pose = true_world.get_pose(true_world.objects[1])
        objb_pose = add_noise(objb_pose)
        experience.append((action, rot, T, objb_pose))

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

        com = np.array(com_particle_dist.particles).T
        print('Mean COM', com@weights, np.diag(np.cov(com_particle_dist.particles, rowvar=False, aweights=weights+1e-3)))
        print('True COM', true_block.com)
        print('Error COM', np.linalg.norm(true_block.com-com@weights))
        estimated_coms.append(com@weights)

        env.disconnect()
        env.cleanup()

    return ParticleDistribution(com_particle_dist.particles, weights), estimated_coms
