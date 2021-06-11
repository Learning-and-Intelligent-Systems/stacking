import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from learning.active.acquire import bald_diagonal_gaussian
from learning.active.utils import ActiveExperimentLogger
from learning.domains.throwing.throwing_data import generate_objects, generate_dataset, label_actions, ParallelDataLoader, xs_to_actions
from learning.domains.throwing.train_latent import get_predictions, train
from learning.models.ensemble import Ensemble
from learning.models.mlp import FeedForward
from learning.models.latent_ensemble import ThrowingLatentEnsemble

def get_latent_ensemble(args):
    n_latents = args.n_objects
    d_observe = 11
    d_latents = 1
    d_pred = 2

    # initialize the LatentEnsemble
    ensemble = Ensemble(base_model=FeedForward,
                        base_args={
                                    'd_in': d_observe + d_latents,
                                    'd_out': d_pred,
                                    'h_dims': [64, 32]
                                  },
                        n_models=args.n_models)
    latent_ensemble = ThrowingLatentEnsemble(ensemble, n_latents=n_latents, d_latents=d_latents)
    if torch.cuda.is_available():
        latent_ensemble = latent_ensemble.cuda()

    return latent_ensemble


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


def active_train(latent_ensemble, dataloader, acquire_fn, logger, args):
    for tx in range(args.max_acquisitions):
        print('Acquisition step', tx)

        # save the current dataset
        logger.save_dataset(dataloader.dataset, tx)

        # train the model on the current dataset
        latent_ensemble.reset()
        latent_ensemble, accs, latents = train(dataloader, dataloader, latent_ensemble, n_epochs=args.n_epochs, return_logs=True)

        # save the ensemble after training
        logger.save_ensemble(latent_ensemble, tx)

        # get new data using the BALD score
        new_data = acquire_fn(latent_ensemble)

        # save teh acquired data
        logger.save_acquisition_data(new_data, None, tx)#new_data, all_samples, tx)

        # add that data to the dataset
        dataloader.add(*new_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-acquisitions',
                        type=int,
                        default=1000,
                        help='Number of iterations to run the main active learning loop for.')
    parser.add_argument('--exp-name', type=str, default='throwing', help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-models', type=int, default=7)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--n-acquire', type=int, default=10)
    parser.add_argument('--n-objects', type=int, default=5)

    args = parser.parse_args()
    args.use_latents = True # hack required for ActiveExperimentLogger compatibility


    objects = generate_objects(args.n_objects)

    # use the sample_action function to get actions, and then preprocess to xs
    data_sampler_fn = lambda n_samples: generate_dataset(objects, args.n_samples, as_tensor=True, label=False)

    # forward pass of the latent ensemble, and marginalize the desired axes
    data_pred_fn = lambda latent_ensemble, unlabeled_data: get_predictions(latent_ensemble,
                                                                           unlabeled_data,
                                                                           n_latent_samples=10,
                                                                           marginalize_latents=True,
                                                                           marginalize_ensemble=False)

    # first two columns of xs are the action params
    data_label_fn = lambda xs, z_ids: label_actions(objects,
                                                    xs_to_actions(xs),
                                                    z_ids,
                                                    as_tensor=True)

    # wrap all that into a function that acquires new data
    acquire_fn = lambda latent_ensemble: acquire_datapoints(latent_ensemble,
                                                            args.n_samples,
                                                            args.n_acquire,
                                                            data_sampler_fn,
                                                            data_pred_fn,
                                                            data_label_fn)


    print('Generating initialization and validation datasets')
    init_dataloader = ParallelDataLoader(TensorDataset(*generate_dataset(objects, 10)),
                                         batch_size=16,
                                         shuffle=True,
                                         n_dataloaders=args.n_models)
    val_dataloader = ParallelDataLoader(TensorDataset(*generate_dataset(objects, 50)),
                                     batch_size=16,
                                     shuffle=True,
                                     n_dataloaders=1)


    # create the latent ensemble
    latent_ensemble = get_latent_ensemble(args)

    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    active_train(latent_ensemble, init_dataloader, acquire_fn, logger, args)
