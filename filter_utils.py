import copy
import numpy as np

from collections import namedtuple
from scipy.stats import multivariate_normal

from block_utils import ParticleDistribution, Environment, Object, World, \
                        Pose, Position, Quaternion, \
                        get_rotated_block, ZERO_POS, ZERO_ROT


'''
:param particles: np.array, N particles each with dimension D (NxD array)
:param weights: np.array, the likelyhood of each particle (N array)
'''

def create_uniform_particles(N, D, ranges):
    '''
    :param N: number of particles
    :param D: number of state dimensions
    :param ranges: list of of length D of (min, max) ranges for each state dimension
    '''
    particles = np.empty((N, D))
    weights = np.ones(N)*(1/N)
    for d in range(D):
        particles[:, d] = np.random.uniform(*ranges[d], size=N)
    return ParticleDistribution(particles, weights)

def create_gaussian_particles(N, D, means, stds):
    '''
    :param N: number of particles
    :param D: number of state dimensions
    :param means: list of of length D of mean for each state dimension
    :param stds: list of of length D of st dev for each state dimension
    '''
    particles = np.empty((N, D))
    for d in range(D):
        particles[:, d] = means[d] + (np.random.randn(N) * stds[d])
    return particles

def sample_particle_distribution(distribution, num_samples=1):
    idxs = np.array(distribution.particles.shape[0])
    idxs = np.random.choice(a=idxs, size=num_samples, replace=True,
        p=distribution.weights)
    return distribution.particles[idxs]

def get_mean(distribution):
    """ Find the mean of a weighted particle distribution

    [description]

    Arguments:
        distribution {ParticleDistribution} -- the input distribution

    Returns:
        np.array -- the mean particle
    """
    # make sure the weights are normalized to sum to 1
    w = distribution.weights / distribution.weights.sum()
    return w @ distribution.particles

def sample_and_wiggle(distribution, experience, obs_model_cov, true_block, ranges=None):
    N, D = distribution.particles.shape
    # NOTE(izzy): note sure if this is an ok way to get the covariance matrix...
    # If the weights has collapsed onto a single particle, then the covariance
    # will collapse too and we won't perturb the particles very much after we
    # sample them. Maybe this should be a uniform covariance with the magnitude
    # being equal to the largest variance?
    # NOTE: (mike): I added a small noise term and a M-H update step which hopefully
    # prevents complete collapse. The M-H update is useful so that we don't sample
    # something completely unlikely by chance.
    # NOTE(izzy): we do not access the COM of true_block. it's just for geometry
    cov = np.cov(distribution.particles, rowvar=False, aweights=distribution.weights+1e-3)
    particles = sample_particle_distribution(distribution, num_samples=N)
    # cov = np.cov(particles, rowvar=False)
    mean = np.mean(particles, axis=0)
    proposed_particles = np.random.multivariate_normal(mean=mean, cov=cov, size=N)

    # Old particles and new particles.
    likelihoods = np.zeros((N,2))
    # Compute likelihood of particles over history so far.
    for action, T, true_pose in experience:
        sim_poses = simulate(np.concatenate([particles, proposed_particles], axis=0),
                             action,
                             T,
                             true_block)
        for ix in range(N):
            likelihoods[ix,0] += np.log(multivariate_normal.pdf(true_pose.pos,
                                                        mean=sim_poses[ix, :],
                                                        cov=obs_model_cov)+1e-8)
            likelihoods[ix,1] += np.log(multivariate_normal.pdf(true_pose.pos,
                                                        mean=sim_poses[N+ix,:],
                                                        cov=obs_model_cov)+1e-8)
    # Calculate M-H acceptance prob.
    prop_probs = np.zeros((N,2))
    for ix in range(N):
        prop_probs[ix,0] = np.log(multivariate_normal.pdf(particles[ix,:], mean=mean, cov=cov)+1e-8)
        prop_probs[ix,1] = np.log(multivariate_normal.pdf(proposed_particles[ix,:], mean=mean, cov=cov)+1e-8)

    p_accept = likelihoods[:,1]+prop_probs[:,0] - (likelihoods[:,0]+prop_probs[:,1])
    accept = np.zeros((N,2))
    accept = np.min(accept, axis=1)

    # Keep particles based on acceptance probability.
    u = np.random.uniform(size=N)
    indices = np.argwhere(u > 1-np.exp(accept)).flatten()
    particles[indices] = proposed_particles[indices]

    # constrain the particles to be within the block if we are given block dimensions
    if ranges is not None:
        for i, (lower, upper) in enumerate(ranges):
            particles[:,i] = np.clip(particles[:,i], lower, upper)

    weights = np.ones(N)/float(N) # weights become uniform again
    return ParticleDistribution(particles, weights)

def simulate(particles, action, T, true_block):
    particle_blocks = [copy.deepcopy(true_block) for particle in particles]
    for (com, particle_block) in zip(particles, particle_blocks):
        particle_block.com = com

    worlds = [make_platform_world(pb, action) for pb in particle_blocks]
    env = Environment(worlds, vis_sim=False)

    for _ in range(T):
        env.step(action=action)

    poses = []
    for particle_world in worlds:
        pose = particle_world.get_pose(particle_world.objects[1])
        poses.append(pose.pos)
    env.disconnect()
    env.cleanup()
    return np.array(poses)


def make_platform_world(p_block, action):
    """ Given a block, create a world that has a platform to push that block off of.
    :param block: The Object which to place on the platform.
    """
    platform = Object.platform()

    p_block.set_pose(Pose(ZERO_POS, Quaternion(*action.rot.as_quat())))
    block = get_rotated_block(p_block)
    block.set_pose(Pose(pos=Position(x=action.pos.x,
                                     y=action.pos.y,
                                     z=platform.dimensions.z+block.dimensions.z/2.),
                        orn=ZERO_ROT))

    return World([platform, block])