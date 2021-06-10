import argparse
from matplotlib import pyplot as pyplot
import numpy as np
import pickle
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from learning.models.latent_ensemble import ThrowingLatentEnsemble


def get_both_loss(latent_ensemble, batches, N, N_samples=10):
    """ compute the loglikelohood of both the latents and the ensemble
    
    Arguments:
        latent_ensemble {ThrowingLatentEnsemble} -- [description]
        batches {list(torch.Tensor)} -- [description]
        N {int} -- total number of training examples
    
    Keyword Arguments:
        N_samples {number} -- number of samples from z (default: {10})
    """

    likelihood_loss = 0
    N_models = latent_ensemble.ensemble.n_models

    for i, batch in enumerate(batches):
        x, z_id, y = val_batches[0]
        N_batch = x.shape[0]
        if torch.cuda.is_available():
            x = x.cuda()
            z_id = z_id.cuda()
            y = y.cuda()

        # run a forward pass of the network and compute the likeliehood of y
        preds = latent_ensemble(x, z_id.long(), ensemble_idx=i, collapse_latents=False, collapse_ensemble=False, N_samples=N_samples)
        D_pred = pred.shape[-1] // 2
        mu, sigma = torch.split(pred, D_pred, dim=-1)
        likelihood_loss += nn.GaussianNLLLoss(y[:, None].expand(N_batch, N_samples), mu, sigma, reduction='sum', full=True)

    likelihood_loss = likelihood_loss/N_models/N_samples

    q_z = torch.distributions.normal.Normal(latent_ensemble.latent_locs, torch.exp(latent_ensemble.latent_logscales))
    p_z = torch.distributions.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
    kl_loss = torch.distributions.kl_divergence(q_z, p_z).sum()

    return (kl_loss + N*likelihood_loss)/N_batch


def evaluate(latent_ensemble, val_dataloader):
    log_prob = 0

    for val_batches in val_dataloader:
        x, z_id, y = val_batches[0]
        if torch.cuda.is_available():
            x = x.cuda()
            z_id = z_id.cuda()
            y = y.cuda()

        # run a forward pass of the network and compute the likeliehood of y
        pred = latent_ensemble(x, z_id.long()).squeeze()
        D_pred = pred.shape[-1] // 2
        mu, sigma = torch.split(pred, D_pred, dim=-1)
        log_prob -= nn.GaussianNLLLoss(y, mu, sigma, reduction='sum', full=True)

    return log_prob


def train(dataloader, val_dataloader, latent_ensemble, n_epochs=30,
    freeze_latents=False,
    freeze_ensemble=False,
    disable_latents=False,
    return_logs=False,
    alternate=False):

    if alternate: raise NotImplementedError()
    if disable_latents: raise NotImplementedError()

    params_optimizer = optim.Adam(latent_ensemble.ensemble.parameters(), lr=1e-3)
    latent_optimizer = optim.Adam([latent_ensemble.latent_locs, latent_ensemble.latent_logscales], lr=1e-3)

    losses = []
    latents = []

    best_weights = None
    best_loss = 1000
    for epoch_idx in range(n_epochs):
        accs = []
        for batch_idx, set_of_batches in enumerate(dataloader):

            if alternate: # take gradient descent steps separately
                batch_loss = 0
                # update the latent distribution while holding the model parameters fixed.
                if (not freeze_latents) and (not disable_latents):
                    latent_optimizer.zero_grad()
                    latent_loss = get_latent_loss(latent_ensemble, set_of_batches[0], N=len(dataloader.loaders[0].dataset))
                    latent_loss.backward()
                    latent_optimizer.step()
                    batch_loss += latent_loss.item()

                # # update the model parameters while sampling from the latent distribution.
                if not freeze_ensemble:
                    params_optimizer.zero_grad()
                    params_loss = get_params_loss(latent_ensemble, set_of_batches, disable_latents, N=len(dataloader.loaders[0].dataset))
                    params_loss.backward()
                    params_optimizer.step()
                    batch_loss += params_loss.item()

            else: # take gradient descent steps together
                params_optimizer.zero_grad()
                latent_optimizer.zero_grad()
                both_loss = get_both_loss(latent_ensemble, set_of_batches, disable_latents, N=len(dataloader.loaders[0].dataset))
                both_loss.backward()
                if (not freeze_latents) and (not disable_latents): latent_optimizer.step()
                if not freeze_ensemble: params_optimizer.step()
                batch_loss = both_loss.item()

            losses.append(batch_loss)

        #TODO: Check for early stopping.
        if val_dataloader is not None:
            val_loss = evaluate(latent_ensemble, val_dataloader)
            print(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(latent_ensemble.state_dict())
                print('New best validation score.')
            print(f'Epoch {epoch_idx}')
 
        latents.append(np.hstack([latent_ensemble.latent_locs.cpu().detach().numpy(),
                                  torch.exp(latent_ensemble.latent_logscales).cpu().detach().numpy()]))

    if val_dataloader is not None:
        latent_ensemble.load_state_dict(best_weights)
    if return_logs:
        return latent_ensemble, losses, latents
    else:
        return latent_ensemble
