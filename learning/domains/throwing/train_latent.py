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

# NOTE(izzy): work in progress -- the following three functions have very similar
# structure, so I wanted to consolidate them in this one function

# def predict(latent_ensemble,
#             batch,
#             n_latent_samples=10,
#             marginalize_latents=True,
#             marginalize_ensemble=True,
#             mode='train',
#             hide_dims=[]):

#         # pull out the input
#         x = batch[0]
#         x = make_x_partially_observable(x, hide_dims)
#         z_id = batch[1]
#         if torch.cuda.is_available():
#             x = x.cuda()
#             z_id = z_id.cuda()

#         with torch.set_grad_enabled(mode=='train'): # conditional "torch.no_grad"
#             pred = latent_ensemble(x, z_id.long(),
#                        collapse_latents=marginalize_latents,
#                        collapse_ensemble=marginalize_ensemble,
#                        N_samples=n_latent_samples).squeeze()

#             if mode == 'predict':
#                 D_pred = pred.shape[-1] // 2
#                 mu, log_sigma = torch.split(pred, D_pred, dim=-1)
#                 sigma = torch.exp(log_sigma)
#                 result = mu, sigma

#             elif mode == 'loss':
#                 # NOTE(izzy): full=True means include the constant terms in the
#                 # log-likelihood computation. In this case, we are
#                 loss_func = nn.GaussianNLLLoss(reduction='sum', full=True)

#                 data_likelihood = loss_func(y[:, None].expand(N_batch, N_samples), mu, torch.exp(log_sigma))


#         return result

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
    total_log_prob = 0
    loss_func = nn.GaussianNLLLoss(reduction='sum', full=True)

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
        total_log_prob += -loss_func(y, mu, torch.exp(log_sigma))

    return total_log_prob / N


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


def generate_or_load_datasets(args):
    # get datasets
    if args.train_dataset == '' and args.val_dataset == '':
        train_objects = generate_objects(args.n_objects)
        print('Generating Training Data')
        train_data_tuple = generate_dataset(train_objects, args.n_train)
        print('Generating Validation Data')
        val_data_tuple = generate_dataset(train_objects, args.n_val)

        if args.save_train_dataset != '':
            print('Saving training data to', args.save_train_dataset)
            with open(args.save_train_dataset, 'wb') as handle:
                pickle.dump(train_data_tuple, handle)
        if args.save_val_dataset != '':
            print('Saving vaidation data to', args.save_val_dataset)
            with open(args.save_val_dataset, 'wb') as handle:
                pickle.dump(val_data_tuple, handle)

    elif args.train_dataset != '' and args.val_dataset != '':
        print('Loading Training Data')
        with open(args.train_dataset, 'rb') as handle:
            train_data_tuple = pickle.load(handle)
        print('Loading Validation Data')
        with open(args.val_dataset, 'rb') as handle:
            val_data_tuple = pickle.load(handle)
        print('Warning: Training and Validation data should use same block set.')

    else:
        print('Both train and val datasets must be specified to load from file.')

    # inialize datasets and dataloaders
    train_dataset = TensorDataset(*train_data_tuple)
    val_dataset = TensorDataset(*val_data_tuple)
    train_dataloader = ParallelDataLoader(dataset=train_dataset,
                                          batch_size=16,
                                          shuffle=True,
                                          n_dataloaders=args.n_models)
    val_dataloader = ParallelDataLoader(dataset=val_dataset,
                                        batch_size=16,
                                        shuffle=False,
                                        n_dataloaders=1)

    return train_dataloader, val_dataloader


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hide-dims', type=str, default='3')
    parser.add_argument('--train-dataset', type=str, default='')
    parser.add_argument('--val-dataset', type=str, default='')
    parser.add_argument('--save-train-dataset', type=str, default='')
    parser.add_argument('--save-val-dataset', type=str, default='')
    parser.add_argument('--save-accs', type=str, default='')
    parser.add_argument('--save-latents', type=str, default='')
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--n-train', type=int, default=500)
    parser.add_argument('--n-val', type=int, default=100)
    parser.add_argument('--n-objects', type=int, default=10)
    parser.add_argument('--n-models', type=int, default=10)
    parser.add_argument('--d-latent', type=int, default=2)

    return parser

def main(args):
    # get the datasets
    train_dataloader, val_dataloader = generate_or_load_datasets(args)

    # variables to define the latent ensemble
    n_latents = args.n_objects
    n_models = args.n_models
    d_latents = args.d_latent
    d_observe = 12
    d_pred = 2
    # produce a list of the dimensions of the object propoerties to make hidden
    hide_dims = [int(d) for d in args.hide_dims.split(',')] if args.hide_dims else []

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
                                           n_epochs=args.n_epochs,
                                           return_logs=True,
                                           hide_dims=hide_dims)

    if args.save_accs != "":
        print('Saving accuracy data to', args.save_accs)
        np.save(args.save_accs, np.array(accs))
    else:
        plt.plot(accs)
        plt.show()

    if args.save_latents != "":
        print('Saving latents data to', args.save_latents)
        np.save(args.save_latents, np.array(latents))


if __name__ == '__main__':
    # get commandline arguments
    parser = get_parser()
    args = parser.parse_args()
    main(args)

