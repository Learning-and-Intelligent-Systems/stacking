import argparse
from learning.models import latent_ensemble
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
import os

from learning.active.utils import ActiveExperimentLogger
from learning.viz_latents import viz_latents


def plot_latent_means(logger):
    means = np.zeros((logger.args.max_acquisitions, 4))
    lows = np.zeros((logger.args.max_acquisitions, 4))
    highs = np.zeros((logger.args.max_acquisitions, 4))
    # go through each acqisition step
    for tx in range(0, logger.args.max_acquisitions):
        print('Eval timestep, ', tx)

        # load the dataset and ensemble from that timestep
        ensemble = logger.get_ensemble(tx)
        means[tx, :] = ensemble.latent_locs[-1, :].detach().numpy()
        sigma = torch.exp(ensemble.latent_logscales[-1, :]).detach().numpy()
        lows[tx, :] = means[tx, :] - sigma 
        highs[tx, :] = means[tx, :] + sigma

    for ix in range(0, 4):
        plt.plot(means[:,ix])
        plt.fill_between(np.arange(0, means.shape[0]), lows[:,ix], highs[:,ix], alpha=0.2)

    plt.xlabel('Acquisition Step')
    plt.ylabel('Latent Value')
    plt.title('Value of each latent variable during fitting')
    plt.savefig(logger.get_figure_path('latent_means.png'))
    plt.clf()


def plot_latent_uncertainty(logger):
    scales = np.zeros((logger.args.max_acquisitions, 4))
    
    # go through each acqisition step
    for tx in range(0, logger.args.max_acquisitions):
        print('Eval timestep, ', tx)

        # load the dataset and ensemble from that timestep
        ensemble = logger.get_ensemble(tx)
        print(ensemble.latent_logscales.shape)
        scales[tx, :] = torch.exp(ensemble.latent_logscales)[-1, :].detach().numpy()
    plt.plot(scales)

    plt.xlabel('Acquisition Step')
    plt.ylabel('Mean Latent Scale')
    plt.title('Variance along each latent dimension')
    plt.savefig(logger.get_figure_path('latent_scale.png'))
    plt.clf()


def plot_latents(logger):
    latents_dir = logger.get_figure_path('latent_images')
    if not os.path.exists(latents_dir):
        os.mkdir(latents_dir)
    
    images = []
    for tx in range(0, logger.args.max_acquisitions, 1):
        ensemble = logger.get_ensemble(tx)
        fname = os.path.join(latents_dir, 'latents_%d.png' % tx)
        with torch.no_grad():
            viz_latents(ensemble.latent_locs, torch.exp(ensemble.latent_logscales), fname=fname)
            images.append(imageio.imread(fname))
    fname = os.path.join(latents_dir, 'latents_evolution.gif')
    imageio.mimsave(fname, images)


def plot_particle_latents(logger):

    means = np.zeros((logger.args.max_acquisitions, 4))
    lows = np.zeros((logger.args.max_acquisitions, 4))
    highs = np.zeros((logger.args.max_acquisitions, 4))
    # go through each acqisition step
    for tx in range(0, logger.args.max_acquisitions):
        print('Eval timestep, ', tx)
        # load the particles for that timestep.
        particles = logger.load_particles(tx)
        #scales.append(torch.exp(ensemble.latent_logscales).mean(axis=0).detach().numpy())
        means[tx, :] = np.mean(particles.particles, axis=0)
        sigma = np.std(particles.particles, axis=0)
        lows[tx, :] = means[tx, :] - sigma 
        highs[tx, :] = means[tx, :] + sigma

    for ix in range(0, 4):
        plt.plot(means[:,ix])
        plt.fill_between(np.arange(0, means.shape[0]), lows[:,ix], highs[:,ix], alpha=0.2)

    plt.xlabel('Acquisition Step')
    plt.ylabel('Latent Value')
    plt.title('Value of each latent variable during fitting')
    plt.savefig(logger.get_figure_path('latent_means.png'))
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    args = parser.parse_args()
    
    logger = ActiveExperimentLogger(args.exp_path, use_latents=True)

    if logger.load_particles(tx=0) is None:
        plot_latents(logger)
        plot_latent_means(logger)
        plot_latent_uncertainty(logger)
    else:
        plot_particle_latents(logger)