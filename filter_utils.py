from collections import namedtuple
import numpy as np
from numpy.random import uniform, randn

ParticleDistribution = namedtuple('ParticleDistribution', 'particles weights')
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
        particles[:, d] = uniform(*ranges[d], size=N)
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
        particles[:, d] = means[d] + (randn(N) * stds[d])
    return particles

def sample_particle_distribution(distribution, num_samples=1):
    idxs = np.random.choice(a=num_samples, size=num_samples, replace=True,
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