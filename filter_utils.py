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

def sample_and_wiggle(distribution, ranges=None):
    N, D = distribution.particles.shape
    # NOTE(izzy): note sure if this is an ok way to get the covariance matrix...
    # If the weights has collapsed onto a single particle, then the covariance
    # will collapse too and we won't perturb the particles very much after we 
    # sample them. Maybe this should be a uniform covariance with the magnitude
    # being equal to the largest variance?
    cov = np.cov(distribution.particles, rowvar=False, aweights=distribution.weights)
    particles = sample_particle_distribution(distribution, num_samples=N)
    # cov = np.cov(particles, rowvar=False)
    particles += np.random.multivariate_normal(mean=np.zeros(D), cov=cov, size=N)

    # constrain the particles to be within the block if we are given block dimensions
    if ranges is not None:
        for i, (lower, upper) in enumerate(ranges):
            particles[:,i] = np.clip(particles[:,i], lower, upper)

    weights = np.ones(N)/float(N) # weights become uniform again
    return ParticleDistribution(particles, weights)
