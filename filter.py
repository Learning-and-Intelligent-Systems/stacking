import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
from mpl_toolkits.mplot3d import Axes3D

covmatrix = np.array([[.003,0,0],[0,.003,0], [0,0,.003]])


from filter_utils import create_uniform_particles
from block_utils import Dimensions, Position, Object, render_objects, get_ps_from_contacts, \
                        Color, Contact

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

def plot_particles(particles, weights):
    #plt.ion()
    fig = plt.figure()
    ax = Axes3D(fig)
    tempmax = max(weights)
    for particle, weight in zip(particles, weights):
        print(weight)
        ax.scatter(*particle, s=100, c=str(1-weight))

    ax.set_xlabel('x') 
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.close()

# make ground truth state
true_com_b = Position(0., 0., 0.)
true_objects = make_objects(true_com_b)

T = 5     # number of steps to simulate per contact state
I = 10      # number of contact states to try

# make particles to estimate the COM of object b
N = 4     # number of particles
D = 3       # dimensions of a single particle

R = .05
com_ranges = [(-true_objects['object_b'].dimensions.width/2, true_objects['object_b'].dimensions.width/2),
                (-true_objects['object_b'].dimensions.length/2, true_objects['object_b'].dimensions.length/2),
                (-true_objects['object_b'].dimensions.height/2, true_objects['object_b'].dimensions.height/2)]
com_particles, weights = create_uniform_particles(N, D, com_ranges)

for i in range(I):
    # select action
    cs = cs_selection(true_objects)
    init_pose = get_ps_from_contacts(cs, true_objects)
    # take action and get observations
    obs_poses = render_objects(true_objects, init_pose, steps=T, vis=True)
    for j in obs_poses:
        gauss_ps(j)
    for t in range(1,T):
        # get observation
        obs_pose = obs_poses[t]

        # update particles
        new_weights = []
        for particle, old_weight in zip(com_particles, weights):
            objects = make_objects(particle)

            # forward simulate particle
            particle_pose = render_objects(objects, obs_poses[t-1], steps=1)
            obj_pose = particle_pose[0]['object_b']
            #new_weight = dist*old_weight if dist < 0.01 else 0.0

            new_weights.append(sc.multivariate_normal.pdf(obs_pose['object_b'], mean=obj_pose,cov=covmatrix) * old_weight)
        weights_sum = sum(new_weights)
        weights += 1.e-300
        weights = np.divide(new_weights, weights_sum)
        print(len(list(filter(lambda w: w > 0, weights))), 'particles remaining')

        plot_particles(com_particles, weights)

        # in the future should resample/redistribute particles
