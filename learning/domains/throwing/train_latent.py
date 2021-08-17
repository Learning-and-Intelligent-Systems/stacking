import argparse
import copy
from matplotlib import pyplot as plt
import numpy as np
import pickle
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from learning.models.ensemble import Ensemble
from learning.models.mlp import FeedForward
from learning.models.latent_ensemble import ThrowingLatentEnsemble
from learning.domains.throwing.throwing_data import generate_objects, generate_dataset, ParallelDataLoader, make_x_partially_observable


def get_predictions(latent_ensemble,
                    unlabeled_data,
                    n_latent_samples=10,
                    marginalize_latents=True,
                    marginalize_ensemble=True,
                    hide_dims=[]):

    dataset = TensorDataset(*unlabeled_data)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=64)

    mus = []
    sigmas = []

    for batch in dataloader:
        x, z_id = batch
        x = make_x_partially_observable(x, hide_dims)
        if torch.cuda.is_available():
            x = x.cuda()
            z_id = z_id.cuda()

        with torch.no_grad():
            # run a forward pass of the network and compute the likeliehood of y
            pred = latent_ensemble(x, z_id.long(),
                                   collapse_latents=marginalize_latents,
                                   collapse_ensemble=marginalize_ensemble,
                                   N_samples=n_latent_samples).squeeze()
            D_pred = pred.shape[-1] // 2
            mu, log_sigma = torch.split(pred, D_pred, dim=-1)
            sigma = torch.exp(log_sigma)

        mus.append(mu)
        sigmas.append(sigma)

    return torch.cat(mus, axis=0), torch.cat(sigmas, axis=0)

def get_both_loss(latent_ensemble,
                  batches,
                  N,
                  N_samples=10,
                  hide_dims=[]):
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
        x, z_id, y = batch
        x = make_x_partially_observable(x, hide_dims)
        N_batch = x.shape[0]

        if torch.cuda.is_available():
            x = x.cuda()
            z_id = z_id.cuda()
            y = y.cuda()

        # run a forward pass of the network and compute the likeliehood of y

        pred = latent_ensemble(x, z_id.long(), ensemble_idx=i, collapse_latents=False, collapse_ensemble=False, N_samples=N_samples)
        D_pred = pred.shape[-1] // 2
        mu, log_sigma = torch.split(pred, D_pred, dim=-1)
        likelihood_loss += loss_func(y[:, None].expand(N_batch, N_samples), mu, torch.exp(log_sigma))

    likelihood_loss = likelihood_loss/N_models/N_samples

    q_z = torch.distributions.normal.Normal(latent_ensemble.latent_locs, torch.exp(latent_ensemble.latent_logscales))
    p_z = torch.distributions.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
    kl_loss = torch.distributions.kl_divergence(q_z, p_z).sum()

    return (kl_loss + N*likelihood_loss)/N_batch


def evaluate(latent_ensemble,
            dataloader,
            hide_dims=[]):
    """ computes the data likelihood

    Arguments:
        latent_ensemble {[type]} -- [description]
        dataloader {[type]} -- [description]

    Keyword Arguments:
        normalized {bool} -- [description] (default: {True})

    Returns:
        [type] -- [description]
    """
    total_prob = 0
    loss_func = nn.GaussianNLLLoss(reduction='none', full=True)

    # decided whether or not to normalize by the amount of data
    N = dataloader.dataset.tensors[0].shape[0]

    for batches in dataloader:
        x, z_id, y = batches[0] if isinstance(dataloader, ParallelDataLoader) else batches
        x = make_x_partially_observable(x, hide_dims)
        if torch.cuda.is_available():
            x = x.cuda()
            z_id = z_id.cuda()
            y = y.cuda()

        # run a forward pass of the network and compute the likeliehood of y
        pred = latent_ensemble(x, z_id.long()).squeeze()
        D_pred = pred.shape[-1] // 2
        mu, log_sigma = torch.split(pred, D_pred, dim=-1)
        total_prob += torch.exp(-loss_func(y, mu, torch.exp(log_sigma))).sum()

    return total_prob / N


def train(dataloader, val_dataloader, latent_ensemble, n_epochs=30,
    freeze_latents=False,
    freeze_ensemble=False,
    return_logs=False,
    hide_dims=[]):

    params_optimizer = optim.Adam(latent_ensemble.ensemble.parameters(), lr=1e-3)
    latent_optimizer = optim.Adam([latent_ensemble.latent_locs, latent_ensemble.latent_logscales], lr=1e-3)

    accs = []
    latents = []
    best_weights = None
    best_acc = 0

    for epoch_idx in range(n_epochs):
        print(f'Epoch {epoch_idx}')

        for batch_idx, set_of_batches in enumerate(dataloader):
            params_optimizer.zero_grad()
            latent_optimizer.zero_grad()
            both_loss = get_both_loss(latent_ensemble, set_of_batches, N=len(dataloader.loaders[0].dataset), hide_dims=hide_dims)
            both_loss.backward()
            if not freeze_latents: latent_optimizer.step()
            if not freeze_ensemble: params_optimizer.step()
            batch_loss = both_loss.item()


        if val_dataloader is not None:
            val_acc = evaluate(latent_ensemble, val_dataloader, hide_dims=hide_dims)
            accs.append(val_acc.item())
            if val_acc > best_acc:
                best_acc = val_acc
                best_weights = copy.deepcopy(latent_ensemble.state_dict())
                print('New best validation score.', val_acc.item())

        latents.append(np.hstack([latent_ensemble.latent_locs.cpu().detach().numpy(),
                                  torch.exp(latent_ensemble.latent_logscales).cpu().detach().numpy()]))

    if val_dataloader is not None:
        latent_ensemble.load_state_dict(best_weights)
    if return_logs:
        return latent_ensemble, accs, latents
    else:
        return latent_ensemble


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--hide-dims', type=str, default='3')

    n_objects = 5
    n_latents = n_objects
    n_models = 10
    d_observe = 12
    d_latents = 3
    d_pred = 2

    args = parser.parse_args()

    # generate training data
    train_objects = generate_objects(n_objects)
    print('Generating Training Data')
    train_dataset = TensorDataset(*generate_dataset(train_objects, 10))
    print('Generating Validation Data')
    val_dataset = TensorDataset(*generate_dataset(train_objects, 10))

    # inialize dataloaders
    train_dataloader = ParallelDataLoader(dataset=train_dataset,
                                          batch_size=16,
                                          shuffle=True,
                                          n_dataloaders=n_models)
    val_dataloader = ParallelDataLoader(dataset=val_dataset,
                                        batch_size=16,
                                        shuffle=False,
                                        n_dataloaders=1)

    # produce a list of the dimensions of the object propoerties to make hidden
    hide_dims = [int(d) for d in args.hide_dims.split(',')]

    # initialize the LatentEnsemble
    ensemble = Ensemble(base_model=FeedForward,
                        base_args={
                                    'd_in': d_observe + d_latents - len(hide_dims),
                                    'd_out': d_pred,
                                    'h_dims': [64, 32]
                                  },
                        n_models=n_models)
    latent_ensemble = ThrowingLatentEnsemble(ensemble, n_latents=n_latents, d_latents=d_latents)
    if torch.cuda.is_available():
        latent_ensemble = latent_ensemble.cuda()


    # train the LatentEnsemble
    latent_ensemble.reset_latents(random=False)
    latent_ensemble, accs, latents = train(train_dataloader,
                                           val_dataloader,
                                           latent_ensemble,
                                           n_epochs=100,
                                           return_logs=True,
                                           hide_dims=hide_dims)

    plt.plot(accs)
    plt.show()