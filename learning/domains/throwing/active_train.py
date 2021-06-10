import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from learning.active.acquire import bald_diagonal_gaussian
from learning.domains.throwing.throwing_data import generate_objects, generate_dataset, label_actions, ParallelDataLoader
from learning.domains.throwing.train_latent import get_predictions, train
from learning.models.ensemble import Ensemble
from learning.models.mlp import FeedForward
from learning.models.latent_ensemble import ThrowingLatentEnsemble

def acquire_datapoints(latent_ensemble,
                       n_samples,
                       n_acquire,
                       data_sampler_fn,
                       data_pred_fn,
                       data_label_fn):

    # sample candidate datapoints
    unlabeled_data = data_sampler_fn(n_samples)

    # compute predictions for each datapoint
    mu, sigma = data_pred_fn(latent_ensemble, unlabeled_data)

    # score the predictions
    scores = bald_diagonal_gaussian(sigma).numpy()

    # choose the best ones
    acquire_indices = np.flip(np.argsort(scores))[:n_acquire]
    xs = torch.Tensor(unlabeled_data[0].numpy()[acquire_indices])
    z_ids = torch.Tensor(unlabeled_data[1].numpy()[acquire_indices])

    # label the acquired data
    ys = data_label_fn(xs, z_ids)

    return xs, z_ids, ys


def active_train(latent_ensemble, dataloader, acquire_fn):
    for acquisition_step in range(10):
        print('Acquisition step', acquisition_step)
        latent_ensemble = train(dataloader, dataloader, latent_ensemble, n_epochs=30)
        new_data = acquire_fn(latent_ensemble)
        dataloader.add(*new_data)


if __name__ == '__main__':
    n_objects = 5
    n_latents = n_objects
    n_models = 10
    n_acquire = 10
    d_observe = 11
    d_latents = 1
    d_pred = 2

    objects = generate_objects(n_objects)

    # use the sample_action function to get actions, and then preprocess to xs
    data_sampler_fn = lambda n_samples: generate_dataset(objects, n_samples, as_tensor=True, label=False)

    # forward pass of the latent ensemble, and marginalize the desired axes
    data_pred_fn = lambda latent_ensemble, unlabeled_data: get_predictions(latent_ensemble,
                                                                           unlabeled_data,
                                                                           n_latent_samples=10,
                                                                           marginalize_latents=True,
                                                                           marginalize_ensemble=False)

    # first two columns of xs are the action params
    data_label_fn = lambda xs, z_ids: label_actions(objects, xs[:,:2], z_ids, as_tensor=True)

    # wrap all that into a function that acquires new data
    acquire_fn = lambda latent_ensemble: acquire_datapoints(latent_ensemble,
                                                            10000,
                                                            n_acquire,
                                                            data_sampler_fn,
                                                            data_pred_fn,
                                                            data_label_fn)

    init_dataloader = ParallelDataLoader(TensorDataset(*generate_dataset(objects, 10)),
                                         batch_size=16,
                                         shuffle=True,
                                         n_dataloaders=n_models)


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

    active_train(latent_ensemble, init_dataloader, acquire_fn)
