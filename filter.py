import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
from mpl_toolkits.mplot3d import Axes3D
from filter_utils import create_uniform_particles
from block_utils import *

covmatrix = 0.003*np.eye(3)

# for now an action is a random contact state
def cs_selection(objects):
    object_a = objects['object_a']
    object_b = objects['object_b']

    p_a_ground = Position(0.,0.,object_a.dimensions.height/2)
    contact_a_ground = Contact('object_a', 'ground', p_a_ground)
    p_x_b_a_mag = (object_a.dimensions.width + object_b.dimensions.width)/2
    p_y_b_a_mag = (object_a.dimensions.length + object_b.dimensions.length)/2
    p_z_b_a = (object_a.dimensions.height + object_b.dimensions.height)/2
    p_b_a = Position(np.random.uniform(-p_x_b_a_mag, p_x_b_a_mag),\
                     np.random.uniform(-p_y_b_a_mag, p_y_b_a_mag),\
                     p_z_b_a)
    contact_b_a = Contact('object_b', 'object_a', p_b_a)
    cs = [contact_a_ground, contact_b_a]
    return cs

# make objects
def make_objects(com_b):
    # get object properties
    com_a = Position(0., 0., 0.)
    dims = Dimensions(.05, .07, .02)
    object_a = Object(dims, 1.0, com_a, Color(1.,1.,0.))
    dims = Dimensions(.02, .04, .01)
    object_b = Object(dims, 1.0, com_b, Color(0.,1.,0.))
    objects = {'object_a':object_a,
               'object_b':object_b}
    return objects

def gauss_ps(objects):
    for i in objects:
        objects[i] = sc.multivariate_normal.rvs(mean=np.array(objects[i]), cov=covmatrix)

plt.ion()
fig = plt.figure()
ax = Axes3D(fig)

def plot_particles(particles, weights):
    for particle, weight in zip(particles, weights):
        print(weight)
        ax.scatter(*particle, s=100, c=str(1-weight))

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
    plt.draw()
    plt.pause(0.1)


# make ground truth state
true_com_b = Position(.0, .0, .0)
true_objects = make_objects(true_com_b)

T = 20     # number of steps to simulate per contact state
I = 10      # number of contact states to try

# make particles to estimate the COM of object b
N = 50     # number of particles
D = 3     # dimensions of a single particle

com_ranges = get_com_ranges(true_objects['object_b'])
com_particles, weights = create_uniform_particles(N, D, com_ranges)

for i in range(I):
    # select action
    cs = cs_selection(true_objects)
    init_pose = get_ps_from_contacts(cs)

    # take action and get noisy observations
    obs_poses = render_objects(true_objects, init_pose, steps=T, vis=True)
    for j in obs_poses:
        gauss_ps(j)

    # forward simulate all particles
    # NOTE(izzy): this will be much faster if you simulate all the particles at
    # the same time. just instantiate a bunch of blocks in one simulator
    # instance, and offset the x,y position of each particle so they can't
    # interact
    particle_poses = {}
    for (pi, particle) in enumerate(com_particles):
        objects = make_objects(particle)
        particle_poses[pi] = render_objects(objects, init_pose, steps=T)

    # update weights for each time step
    for t in range(1,T):
        # get observation
        obs_pose = obs_poses[t]

        # update particles
        new_weights = []
        for pi, (particle, old_weight) in enumerate(zip(com_particles, weights)):
            particle_pose = particle_poses[pi][t]
            new_weights.append(sc.multivariate_normal.pdf(obs_pose['object_b'],
                                                        mean=particle_pose['object_b'],
                                                        cov=covmatrix) * old_weight)

        # normalize particle weights
        weights_sum = sum(new_weights)
        weights = np.divide(new_weights, weights_sum)
        print('max particle weight: ', max(weights))

        # visualize particles
        plot_particles(com_particles, weights)

        # in the future should resample/redistribute particles
