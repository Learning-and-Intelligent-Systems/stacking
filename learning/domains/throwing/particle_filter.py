""" This training loop is supposed to mirror the one in
learning.domains.throwing.train_latent.train In this case, we do
not use gradient descent to update the latents, but rather particle
filtering. """
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn

from learning.domains.throwing.train_latent import get_predictions
from learning.domains.throwing.throwing_data import make_x_partially_observable


def get_particle_likelihoods(dataloader, latent_ensemble, hide_dims):
    """ iterate through the dataloader and compute the likelihood of each
    particle on the data """

    latent_ensemble.fitting = True # don't sample from the particles
    loss_func = nn.GaussianNLLLoss(reduction='none', full=True)
    N_latents = latent_ensemble.n_latents
    N_particles = latent_ensemble.n_particles
    particle_likelihoods = np.zeros([N_latents, N_particles])

    with torch.no_grad():

        for batch_idx, set_of_batches in enumerate(dataloader):
            # we take in a ParallelDataloader to match the typical
            # train_latent.train spec, but we don't need all the
            # parallel batches
            x, z_id, y = set_of_batches[0]
            x = make_x_partially_observable(x, hide_dims)
            N_batch = x.shape[0]

            if torch.cuda.is_available():
                x = x.cuda()
                z_id = z_id.cuda()
                y = y.cuda()

            # pred should be [N_batch x N_particles x 2*D_output]
            pred = latent_ensemble(x, z_id.long(),
                                   N_samples=N_particles,
                                   collapse_latents=False,
                                   collapse_ensemble=True).squeeze(dim=1)
            D_pred = pred.shape[-1] // 2
            mu, log_sigma = torch.split(pred, D_pred, dim=-1)
            sigma = torch.exp(log_sigma)

            # compute likelihood of each particle on the batch data
            # NOTE(izzy): even tho I have reduction='none' set for
            # loss_func, it still seems to sum along all but the first
            # dimension, so i just made the particle the first dimension
            batch_likelihoods = -loss_func(y[:, None, None].expand(N_batch, N_particles, 1),
                                               mu, sigma).sum(axis=-1)

            # agregate likelihood into the particle set for each object
            particle_likelihoods[z_id.long()] += batch_likelihoods.squeeze().numpy()

    latent_ensemble.fitting = False # we'll sample from particles in the future
    return particle_likelihoods

def resample(particles, weights):
    N, D = particles.shape

    # choose each particle a number of times proportional to its weight
    idxs = np.random.choice(a=np.array(N), size=N, replace=True,
        p=weights/weights.sum())
    resampled_particles = particles[idxs]

    # move the particles slightly
    return resampled_particles + 0.05 * np.random.randn(*resampled_particles.shape)

    # TODO(izzy): implement a Metropolis Hastings update
    # propose new particles via a gaussian fit to the old distribution
    # mean = np.mean(resampled_particles, axis=0)
    # cov = np.cov(resampled_particles, rowvar=False) + np.eye(D)*0.5
    # proposed_particles = np.random.multivariate_normal(mean=mean, cov=cov, size=N)

    

def update_particle_filter(dataloader, val_dataloader, latent_ensemble, n_epochs=30,
    freeze_latents=False,
    freeze_ensemble=False,
    return_logs=False,
    hide_dims=[3]):

    # TODO(izzy): add logging as needed
    latents = []
    accs = []

    with torch.no_grad():
        for epoch_idx in range(n_epochs):
            print(f'Epoch {epoch_idx}')
            particle_likelihoods = get_particle_likelihoods(dataloader, latent_ensemble, hide_dims)
     
            # resample the each latent variable individually
            for i in range(latent_ensemble.n_latents):
                particles = latent_ensemble.latent_locs[i].detach().numpy()
                weights = np.exp(particle_likelihoods[i])
                new_particles = resample(particles, weights)
                latent_ensemble.latent_locs[i] = torch.Tensor(new_particles)

            #     plt.hist(new_particles, alpha=0.3)
            # plt.show()


    if val_dataloader is not None:
        latent_ensemble.load_state_dict(best_weights)
    if return_logs:
        return latent_ensemble, accs, latents
    else:
        return latent_ensemble