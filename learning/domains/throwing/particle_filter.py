""" This training loop is supposed to mirror the one in
learning.domains.throwing.train_latent.train In this case, we do
not use gradient descent to update the latents, but rather particle
filtering. """

import numpy as np
import torch
from torch import nn

from learning.domains.throwing.train_latent import get_predictions
from learning.domains.throwing.throwing_data import make_x_partially_observable



def update_particle_filter(dataloader, val_dataloader, latent_ensemble, n_epochs=30,
    freeze_latents=False,
    freeze_ensemble=False,
    return_logs=False,
    hide_dims=[3]):

    accs = []
    latents = []
    best_weights = None
    best_acc = -np.inf
    N_particles = latent_ensemble.n_particles
    latent_ensemble.fitting = True # don't sample from the particles

    loss_func = nn.GaussianNLLLoss(reduction='none', full=True)

    with torch.no_grad():
        for epoch_idx in range(n_epochs):
            print(f'Epoch {epoch_idx}')
            particle_likelihoods = np.zeros(N_particles)
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
                                       collapse_ensemble=True).squeeze()
                D_pred = pred.shape[-1] // 2
                mu, log_sigma = torch.split(pred, D_pred, dim=-1)
                sigma = torch.exp(log_sigma)

                # compute likelihood of each particle on the batch data
                # NOTE(izzy): even tho I have reduction='none' set for
                # loss_func, it still seems to sum along all but the first
                # dimension, so i just made the particle the first dimension
                batch_likelihoods = -loss_func(y[None, :].expand(N_particles, N_batch),
                                               torch.swapaxes(mu, 0, 1),
                                               torch.swapaxes(sigma, 0, 1))
                # agregate likelihood
                particle_likelihoods = batch_likelihoods

    latent_ensemble.fitting = False # we'll sample from particles in the future

    if val_dataloader is not None:
        latent_ensemble.load_state_dict(best_weights)
    if return_logs:
        return latent_ensemble, accs, latents
    else:
        return latent_ensemble