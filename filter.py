import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from filter_utils import create_uniform_particles
from block_utils import Environment, Object, Position, Dimensions, World, \
                        Pose, Orientation, Color, get_com_ranges, Contact, \
                        get_ps_from_contacts

true_obs_cov = 0.00004*np.eye(3)
obs_model_cov = 0.00004*np.eye(3)

# for now an action is a random contact state
def cs_selection(world):
    object_a = world.objects[0]
    object_b = world.objects[1]

    p_a_ground = Position(0.,0.,object_a.dimensions.z/2)
    contact_a_ground = Contact('object_a', 'ground', p_a_ground)
    p_x_b_a_mag = (object_a.dimensions.x + object_b.dimensions.x)/2
    p_y_b_a_mag = (object_a.dimensions.y + object_b.dimensions.y)/2
    p_z_b_a = (object_a.dimensions.z + object_b.dimensions.z)/2
    p_b_a = Position(np.random.uniform(-p_x_b_a_mag, p_x_b_a_mag),
                     np.random.uniform(-p_y_b_a_mag, p_y_b_a_mag),
                     p_z_b_a)
    contact_b_a = Contact('object_b', 'object_a', p_b_a)
    cs = [contact_a_ground, contact_b_a]
    return cs

# make objects
def make_world(com_b):
    # get object properties
    mass = 1.0
    com_a = Position(0., 0., 0.)
    dims = Dimensions(.05, .07, .02)
    object_a = Object('object_a', dims, mass, com_a, Color(1.,1.,0.))
    dims = Dimensions(.02, .04, .01)
    object_b = Object('object_b', dims, mass, com_b, Color(0.,1.,0.))
    world = World([object_a, object_b])
    return world

def add_noise(pose):
    noisy_pos = Position(*multivariate_normal.rvs(mean=pose.pos, cov=true_obs_cov))
    return Pose(noisy_pos, pose.orn)

plt.ion()
fig = plt.figure()
ax = Axes3D(fig)

def plot_particles(particles, weights, t=None):
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
    ax.view_init(elev=100., azim=0.0)  # top down view of x-y plane
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.draw()
    plt.pause(0.1)

# make ground truth world
true_com_b = Position(.0, .0, .0)
true_world = make_world(true_com_b)

T = 15     # number of steps to simulate per contact state
I = 10      # number of contact states to try

# make particles to estimate the COM of object b
N = 300     # number of particles
D = 3     # dimensions of a single particle

# create particle worlds
com_ranges = get_com_ranges(true_world.objects[1])
com_particles, weights = create_uniform_particles(N, D, com_ranges)
particle_worlds = [make_world(particle) for particle in com_particles]

for i in range(I):
    # select contact state and calculate pose
    cs = cs_selection(true_world)
    init_pose = get_ps_from_contacts(cs)

    # set object poses in all worlds
    true_world.set_poses(init_pose)
    for particle_world in particle_worlds:
        particle_world.set_poses(init_pose)

    # create pyBullet environment for true world and particle worlds
    env = Environment(particle_worlds+[true_world], vis_sim=False)

    for t in range(T):
        # step all worlds
        env.step(vis_frames=False)

        # get ground truth object_b pose (observation)
        objb_pose = true_world.get_pose(true_world.objects[1])
        objb_pose = add_noise(objb_pose)

        # update all particle weights
        new_weights = []

        for pi, (particle_world, old_weight) in enumerate(zip(particle_worlds, weights)):
            particle_objb_pose = particle_world.get_pose(particle_world.objects[1])
            obs_model = multivariate_normal.pdf(objb_pose.pos, mean=particle_objb_pose.pos, cov=obs_model_cov)
            new_weight = old_weight * obs_model
            new_weights.append(new_weight)

        # normalize particle weights
        weights_sum = sum(new_weights)
        weights = np.divide(new_weights, weights_sum)
        print('max particle weight: ', max(weights))

        # visualize particles
        plot_particles(com_particles, weights, t=t)


    env.disconnect()

# remove temp urdf files (they will accumulate quickly)
shutil.rmtree('tmp_urdfs')
