import argparse
from learning.models import latent_ensemble
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
import os
from matplotlib import pyplot as plt

from block_utils import World, Environment, Object
from learning.active.utils import ActiveExperimentLogger
from learning.evaluate.active_evaluate_towers import get_predictions_with_particles
from learning.viz_latents import viz_latents
from tamp.misc import get_train_and_fit_blocks


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


def visualize_fitted_block(logger, n_towers=100):
    """ Generates images analyzing the behavior of the block as it moves over an 
    arbitrary block below it.

    """
    latent_ensemble = logger.get_ensemble(logger.args.max_acquisitions-1)
    if torch.cuda.is_available():
        latent_ensemble = latent_ensemble.cuda()
    particles = logger.load_particles(logger.args.max_acquisitions-1)
    top_block_ix = -1

    block_set = get_train_and_fit_blocks(pretrained_ensemble_path=logger.args.pretrained_ensemble_exp_path,
                                         use_latents=True,
                                         fit_blocks_fname=logger.args.block_set_fname,
                                         fit_block_ixs=logger.args.eval_block_ixs)
    
    bottom_block = block_set[0]
    top_block = block_set[top_block_ix]
    top_block.pose

    template_tower = np.stack([bottom_block.vectorize(),
                               top_block.vectorize()])
    template_tower[1, 9] = (top_block.dimensions.z + bottom_block.dimensions.z)/2.

    leftmost = -(top_block.dimensions.x - bottom_block.dimensions.x)/2.
    rightmost = (top_block.dimensions.x + bottom_block.dimensions.x)/2.
    top_xs = np.linspace(leftmost, rightmost, n_towers)

    if False:
        for x in top_xs:
            template_tower[1, 7] = x
            block_tower = [Object.from_vector(template_tower[bx, :]) for bx in range(2)]
            w = World(block_tower)
            env = Environment([w], vis_sim=True, vis_frames=True)
            input('Next?')
            env.disconnect()

    # Build a dataset.
    towers = np.stack([template_tower for _ in range(n_towers)])
    towers[:, 1, 7] = top_xs
    block_ids = np.stack([0, len(block_set)-1] for _ in range(n_towers))
    labels = np.array([0] * n_towers)
    tower_dict = {}
    tower_dict['2block'] = {
        'towers': towers,
        'block_ids': block_ids,
        'labels': labels
    }
    print(towers.shape, block_ids.shape, labels.shape)
    

    #particles.particles[:, 0] = -20.
    #print(particles.particles)
    preds = get_predictions_with_particles(particles=particles.particles[0:50], 
                                           observation=tower_dict, 
                                           ensemble=latent_ensemble, 
                                           fitting_block_ix=len(block_set)-1).cpu().numpy()
    
    actual_x = bottom_block.dimensions.x/2 - top_block.com.x
    
    #print(bottom_block.dimensions, top_block.com)
    plt.plot([actual_x, actual_x], [0, 1])
    plt.plot(top_xs, preds)
    plt.ylim(0, 1.1)
    plt.show()



    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    args = parser.parse_args()
    
    logger = ActiveExperimentLogger(args.exp_path, use_latents=True)
    visualize_fitted_block(logger)
    sys.exit()
    if logger.load_particles(tx=0) is None:
        plot_latents(logger)
        plot_latent_means(logger)
        plot_latent_uncertainty(logger)
    else:
        plot_particle_latents(logger)