import argparse
import copy
from matplotlib import pyplot as pyplot
import numpy as np
import pickle
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from learning.models.ensemble import Ensemble
from learning.models.mlp import FeedForward
from learning.models.latent_ensemble import ThrowingLatentEnsemble
from learning.domains.throwing.generate_data import generate_objects, generate_dataset
from learning.domains.towers.tower_data import ParallelDataLoader


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
    loss_func = nn.GaussianNLLLoss(reduction='sum', full=True)

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
        mu, log_sigma = torch.split(pred, D_pred, dim=-1)
        print(mu, log_sigma)
        likelihood_loss += loss_func(y[:, None].expand(N_batch, N_samples), mu, torch.exp(log_sigma))
        print(likelihood_loss)
    likelihood_loss = likelihood_loss/N_models/N_samples

    q_z = torch.distributions.normal.Normal(latent_ensemble.latent_locs, torch.exp(latent_ensemble.latent_logscales))
    p_z = torch.distributions.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
    kl_loss = torch.distributions.kl_divergence(q_z, p_z).sum()

    return (kl_loss + N*likelihood_loss)/N_batch


def evaluate(latent_ensemble, val_dataloader):
    log_prob = 0
    loss_func = nn.GaussianNLLLoss(reduction='sum', full=True)

    for val_batches in val_dataloader:
        x, z_id, y = val_batches[0]
        if torch.cuda.is_available():
            x = x.cuda()
            z_id = z_id.cuda()
            y = y.cuda()

        # run a forward pass of the network and compute the likeliehood of y
        pred = latent_ensemble(x, z_id.long()).squeeze()
        D_pred = pred.shape[-1] // 2
        mu, log_sigma = torch.split(pred, D_pred, dim=-1)
        log_prob -= loss_func(y, mu, torch.exp(log_sigma))

    return log_prob


def train(dataloader, val_dataloader, latent_ensemble, n_epochs=30,
    freeze_latents=False,
    freeze_ensemble=False,
    return_logs=False,):

    params_optimizer = optim.Adam(latent_ensemble.ensemble.parameters(), lr=1e-3)
    latent_optimizer = optim.Adam([latent_ensemble.latent_locs, latent_ensemble.latent_logscales], lr=1e-3)

    losses = []
    latents = []

    best_weights = None
    best_loss = 1000
    for epoch_idx in range(n_epochs):
        print(f'Epoch {epoch_idx}')
        accs = []
        for batch_idx, set_of_batches in enumerate(dataloader):
            print(set_of_batches)
            params_optimizer.zero_grad()
            latent_optimizer.zero_grad()
            both_loss = get_both_loss(latent_ensemble, set_of_batches, disable_latents, N=len(dataloader.loaders[0].dataset))
            both_loss.backward()
            if not freeze_latents: latent_optimizer.step()
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

        latents.append(np.hstack([latent_ensemble.latent_locs.cpu().detach().numpy(),
                                  torch.exp(latent_ensemble.latent_logscales).cpu().detach().numpy()]))

    if val_dataloader is not None:
        latent_ensemble.load_state_dict(best_weights)
    if return_logs:
        return latent_ensemble, losses, latents
    else:
        return latent_ensemble


if __name__ == '__main__':
    n_objects = 10
    n_latents = n_objects
    n_models = 10
    d_observe = 6
    d_latents = 4
    d_pred = 2


    # generate training data
    train_objects = generate_objects(n_objects)
    train_dataset = TensorDataset(*generate_dataset(train_objects, 50))
    val_dataset = TensorDataset(*generate_dataset(train_objects, 50))

    # inialize dataloaders
    train_dataloader = ParallelDataLoader(dataset=train_dataset,
                                          batch_size=16,
                                          shuffle=True,
                                          n_dataloaders=n_models)
    val_dataloader = ParallelDataLoader(dataset=val_dataset,
                                        batch_size=16,
                                        shuffle=False,
                                        n_dataloaders=1)


    # initialize the LatentEnsemble
    ensemble = Ensemble(base_model=FeedForward,
                        base_args={
                                    'd_in': d_observe + d_latents,
                                    'd_out': d_pred,
                                    'h_dims': [64, 32]
                                  },
                        n_models=n_models)
    latent_ensemble = ThrowingLatentEnsemble(ensemble, n_latents=n_latents, d_latents=d_latents)
    if torch.cuda.is_available():
        latent_ensemble = latent_ensemble.cuda()

    # train the LatentEnsemble
    latent_ensemble.reset_latents(random=False)
    latent_ensemble, losses, latents = train(train_dataloader,
                                             val_dataloader,
                                             latent_ensemble,
                                             n_epochs=30,
                                             return_logs=True)