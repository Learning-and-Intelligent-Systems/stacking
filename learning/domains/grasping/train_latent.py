import argparse
from matplotlib import pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.spatial.transform import Rotation
import torch
import copy
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from learning.active.utils import ActiveExperimentLogger
from learning.domains.towers.tower_data import TowerDataset, ParallelDataLoader
from learning.models.gn import FCGN
from learning.models.ensemble import Ensemble
from learning.models.latent_ensemble import LatentEnsemble
from learning.viz_latents import viz_latents


def get_both_loss(latent_ensemble, batches, disable_latents, N, N_samples=10):
    likelihood_loss = 0
    N_models = latent_ensemble.ensemble.n_models

    for i, batch in enumerate(batches): # for each model === for each sample from the model distribution
        grasps, object_ids, labels = batch
        N_batch = grasps.shape[0]

        if torch.cuda.is_available():
            grasps = grasps.cuda()
            object_ids = object_ids.cuda()
            labels = labels.cuda()

        if disable_latents:
            preds = latent_ensemble.ensemble.models[i].forward(grasps).squeeze()
        else:
            preds = latent_ensemble(grasps[:,:-5,:], object_ids.long(), ensemble_idx=i, collapse_latents=False, collapse_ensemble=False, N_samples=N_samples).squeeze()
        
        total_samples = preds.shape[1]
        likelihood_loss += F.binary_cross_entropy(preds, labels[:, None].expand(N_batch, total_samples), reduction='sum')

    likelihood_loss = likelihood_loss/N_models/total_samples

    q_z = torch.distributions.normal.Normal(latent_ensemble.latent_locs, torch.exp(latent_ensemble.latent_logscales))
    p_z = torch.distributions.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
    kl_loss = torch.distributions.kl_divergence(q_z, p_z).sum()

    return (kl_loss + N*likelihood_loss)/N_batch


def get_params_loss(latent_ensemble, batches, disable_latents, N):
    """
    1. sample ~ latents
    2. samples -(model)-> likelihood
    3. gradient descent step on model params

    Arguments:
        model {[type]} -- [description]
        latents {[type]} -- [description]
        batches {[type]} -- note that each model in the ensemble gets its own batch
    """

    likelihood_loss = 0
    for i, batch in enumerate(batches): # for each model === for each sample from the model distribution
        towers, block_ids, labels = batch
        if torch.cuda.is_available():
            towers = towers.cuda()
            block_ids = block_ids.cuda()
            labels = labels.cuda()
        # TODO(izzy): I'm dropping the first four elements from the vectorized
        # towers, mass and COM xyz. I'm not sure if this is the best place to
        # do that because it it is still in the datast. It should probably be a
        # flag in the TowerDataset?
        if disable_latents:
            preds = latent_ensemble.ensemble.models[i].forward(towers).squeeze()
        else:
            preds = latent_ensemble(towers[:,:-5,:], block_ids.long(), collapse_latents=True, collapse_ensemble=False, ensemble_idx=i)
        likelihood_loss += F.binary_cross_entropy(preds.squeeze(), labels.squeeze(), reduction='sum')

    # we sum the likelihoods for every input in the batch, but we want the
    # expected likelihood under the ensemble which means we take the mean
    return N*likelihood_loss/towers.shape[0]/latent_ensemble.ensemble.n_models


def get_latent_loss(latent_ensemble, batch, N, N_samples=10):
    """
    [mu, sigma] -(reparam)-> [sample] -(thru ensemble)-> [likelihood]
    [mu, sigma] -> [KL]

    gradient descent step on model params [likelihood + KL]

    Choices:
        * either sample a single model ~ ensemble
        * take mean of likelihood under ensemble

    Arguments:
        ensemble {[type]} -- [description]
        latents {[type]} -- [description]
        batch {[type]} -- [description]
    """
    towers, block_ids, labels = batch
    if torch.cuda.is_available():
        towers = towers.cuda()
        block_ids = block_ids.cuda()
        labels = labels.float().cuda()
    # NOTE(izzy): we update the params of the latent distribution using the
    # reparametrization technique through a sample from that distribution. we may
    # wish to draw multiple samples from the latent distribution to reduce the
    # variance of the updates
    # TODO(izzy): I'm dropping the first four elements from the vectorized
    # towers, mass and COM xyz. I'm not sure if this is the best place to
    # do that because it it is still in the datast. It should probably be a
    # flag in the TowerDataset?

    # old version: takes the mean of the labels before computing likelihood
    # preds = latent_ensemble(towers[:,:,4:], block_ids.long(), collapse_latents=True, collapse_ensemble=True)#, np.random.randint(0, len(latent_ensemble.ensemble.models))) # take the mean of the ensemble
    # likelihood_loss = F.binary_cross_entropy(preds.squeeze(), labels.squeeze(), reduction='sum')
    # and compute the kl divergence

    # updated version June 3: we want to take the expectation outside the likelihood
    preds = latent_ensemble(towers[:,:-5,:], block_ids.long(), collapse_latents=False, collapse_ensemble=True, N_samples=N_samples)
    likelihood_losses = F.binary_cross_entropy(preds.squeeze(), labels[:, None].expand(towers.shape[0], N_samples), reduction='none')
    likelihood_loss = likelihood_losses.mean(axis=1).sum(axis=0)

    # Option 1: Calculate KL for every latent in each batch.
    q_z = torch.distributions.normal.Normal(latent_ensemble.latent_locs, torch.exp(latent_ensemble.latent_logscales))
    p_z = torch.distributions.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))

    kl_loss = torch.distributions.kl_divergence(q_z, p_z).sum()

    return (kl_loss + N*likelihood_loss)/towers.shape[0]

def train(dataloader, val_dataloader, latent_ensemble, n_epochs=30,
    freeze_latents=False,
    freeze_ensemble=False,
    disable_latents=False,
    return_logs=False,
    alternate=False,
    args=None,
    show_epochs=False):

    params_optimizer = optim.Adam(latent_ensemble.ensemble.parameters(), lr=1e-3)
    latent_optimizer = optim.Adam([latent_ensemble.latent_locs, latent_ensemble.latent_logscales], lr=1e-2)

    losses = []
    latents = []
   
    best_weights = None
    best_loss = 1000
    for epoch_idx in range(n_epochs):
        if show_epochs:
            print('Epoch', epoch_idx)
        accs = []
        latent_ensemble.ensemble.train()
        for batch_idx, set_of_batches in enumerate(dataloader):            
            if alternate: # take gradient descent steps separately
                batch_loss = 0
                # update the latent distribution while holding the model parameters fixed.
                if (not freeze_latents) and (not disable_latents):
                    latent_optimizer.zero_grad()
                    latent_loss = get_latent_loss(latent_ensemble, set_of_batches[0], N=len(dataloader.loaders[0].dataset))
                    latent_loss.backward()
                    # If we are only updating the new latents, zero out gradients for the training blocks.
                    if args.fit:
                        latent_ensemble.latent_locs.grad.data[:args.num_train_blocks, :].fill_(0.)
                        latent_ensemble.latent_logscales.grad.data[:args.num_train_blocks, :].fill_(0.)

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
                if args.fit:
                    latent_ensemble.latent_locs.grad.data[:args.num_train_blocks, :].fill_(0.)
                    latent_ensemble.latent_logscales.grad.data[:args.num_train_blocks, :].fill_(0.)

                if (not freeze_latents) and (not disable_latents): latent_optimizer.step()
                if not freeze_ensemble: params_optimizer.step()
                batch_loss = both_loss.item()
            losses.append(batch_loss)

        #TODO: Check for early stopping.
        if val_dataloader is not None:
            val_loss = evaluate(latent_ensemble, val_dataloader, disable_latents=disable_latents)
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(latent_ensemble.state_dict())
                print('Saved.')
            print(f'Epoch {epoch_idx}')
            print('Train Accuracy:')
            print(compute_accuracies(latent_ensemble, dataloader, disable_latents=disable_latents))
            print('Val Accuracy:')
            print(compute_accuracies(latent_ensemble, val_dataloader, disable_latents=disable_latents))
        print('LOCS:')        
        print(latent_ensemble.latent_locs[:10, :])
        print('SCALES:')
        print(latent_ensemble.latent_logscales[:10, :])
        latents.append(np.hstack([latent_ensemble.latent_locs.cpu().detach().numpy(),
                                  torch.exp(latent_ensemble.latent_logscales).cpu().detach().numpy()]))

    if val_dataloader is not None:
        latent_ensemble.load_state_dict(best_weights)
    if return_logs:
        return latent_ensemble, losses, latents
    else:
        return latent_ensemble

    # Note (Mike): When doing active learning, add new towers to train_dataset (not train_loader).

def evaluate(latent_ensemble, data_loader, disable_latents, val_metric='loss'):
    acc = []
    losses = []

    preds = []
    labels = []
    latent_ensemble.ensemble.eval()
    for val_batches in data_loader:
        towers, block_ids, label = val_batches[0]
        if torch.cuda.is_available():
            grasps = towers.cuda()
            object_ids = block_ids.cuda()
            label = label.cuda()
        if disable_latents:
            pred = latent_ensemble.ensemble.forward(grasps).squeeze()
        else:
            pred = latent_ensemble(grasps[:,:-5,:], object_ids.long()).squeeze()
        if len(pred.shape) == 0: pred = pred.unsqueeze(-1)
        loss = F.binary_cross_entropy(pred, label)
        losses.append(loss.item())
        with torch.no_grad():
            preds += (pred > 0.5).cpu().float().numpy().tolist()
            labels += label.cpu().numpy().tolist()
    if val_metric == 'loss':
        score = np.mean(losses)
    else:
        score = -f1_score(labels, preds)

    return score

def compute_accuracies(latent_ensemble, data_loader, disable_latents):
    latent_ensemble.ensemble.eval()
    with torch.no_grad():
        predictions, labels = [], []
        for val_batches in data_loader:
            grasps, object_ids, y = val_batches[0]
            if torch.cuda.is_available():
                grasps = grasps.cuda()

            if disable_latents:
                probs = latent_ensemble.ensemble.forward(grasps).squeeze()
            else:
                probs = latent_ensemble(grasps[:,:-5,:], object_ids.long()).squeeze()
            preds = (probs > 0.5).float().cpu()
            
            predictions.append(preds)
            labels.append(y)

        predictions = torch.cat(predictions).numpy()
        labels = torch.cat(labels).numpy()

        acc = accuracy_score(labels, predictions)

    return acc

def test(latent_ensemble, train_loader, test_loader, disable_latents, n_epochs=50):
    latent_ensemble.reset_latents(random=False)

    print('Test Accuracy with prior latents:')
    for k, v in compute_accuracies(latent_ensemble, test_loader, disable_latents=disable_latents).items():
        print(k, '%.4f' % np.mean(v))
    # print(latent_ensemble.latent_locs, latent_ensemble.latent_scales)

    # estimate the latents for the test data, but without updating the model
    # parameters
    latent_ensemble, losses, latents = train(train_loader, None, latent_ensemble, n_epochs=n_epochs, freeze_ensemble=True, disable_latents=disable_latents, return_logs=True)
    with torch.no_grad():
        viz_latents(latent_ensemble.latent_locs.cpu().detach(), torch.exp(latent_ensemble.latent_logscales).cpu().detach())
    # np.save('learning/experiments/logs/latents/fit_during_test.npy', latents)

    print('Test Accuracy with posterior latents:')
    for k, v in compute_accuracies(latent_ensemble, test_loader, disable_latents=disable_latents).items():
        print(k, '%.4f' % np.mean(v))
    # print(latent_ensemble.latent_locs, latent_ensemble.latent_scales)
