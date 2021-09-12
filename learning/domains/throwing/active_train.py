import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from learning.active.acquire import bald_diagonal_gaussian
from learning.active.utils import ActiveExperimentLogger
from learning.domains.throwing.particle_filter import update_particle_filter
from learning.domains.throwing.throwing_data import generate_objects, generate_dataset, label_actions, ParallelDataLoader, xs_to_actions
from learning.domains.throwing.train_latent import get_predictions, train
from learning.models.ensemble import Ensemble
from learning.models.mlp import FeedForward
from learning.models.latent_ensemble import ThrowingLatentEnsemble, convert_to_particle_filter_latent_ensemble

def get_latent_ensemble(args):
    n_latents = args.n_objects
    d_observe = 12
    d_latents = 1
    d_pred = 2
    hide_dims = [int(d) for d in args.hide_dims.split(',')] if args.hide_dims else []

    if args.fitting:
        # if we are fitting latents, then we load the latent ensemble from a previous exp-path
        assert(args.use_latents and len(args.latent_ensemble_exp_path) > 0 and args.latent_ensemble_tx >= 0)
        logger = ActiveExperimentLogger.get_experiments_logger(args.latent_ensemble_exp_path, args)
        logger.args.throwing = True # hack to get it to load a ThrowingLatentEnsemble
        latent_ensemble = logger.get_ensemble(args.latent_ensemble_tx)
        # change the number of objects we fit to (also resets latents)
        latent_ensemble = latent_ensemble.change_number_of_latents(n_latents)
        # convert to particle filtering version if needed
        if args.use_particle_filter:
            latent_ensemble = convert_to_particle_filter_latent_ensemble(latent_ensemble)

    else:
        # if we are fitting the model, then we create a new latent ensemble
        ensemble = Ensemble(base_model=FeedForward,
                            base_args={
                                        'd_in': d_observe + d_latents - len(hide_dims),
                                        'd_out': d_pred,
                                        'h_dims': [64, 32]
                                      },
                            n_models=args.n_models)
        latent_ensemble = ThrowingLatentEnsemble(ensemble,
                                                 n_latents=n_latents,
                                                 d_latents=d_latents)
                                                 #disable_latents=not args.use_latents)

    if torch.cuda.is_available():
        latent_ensemble = latent_ensemble.cuda()

    return latent_ensemble


def acquire_datapoints(latent_ensemble,
                       n_samples,
                       n_acquire,
                       data_sampler_fn,
                       data_pred_fn,
                       data_label_fn,
                       acquisition='bald'):

    # sample candidate datapoints
    unlabeled_data = data_sampler_fn(n_samples)

    if acquisition == 'bald':
        # compute predictions for each datapoint
        mu, sigma = data_pred_fn(latent_ensemble, unlabeled_data)
        # score the predictions
        scores = bald_diagonal_gaussian(mu, sigma).numpy()

    elif acquisition == 'random':
        # random scores
        scores = np.random.rand(unlabeled_data[0].shape[0])

    else:
        raise NotImplementedError(f"Unknown acquisition strategy: {acquisition}.")

    # choose the best ones
    acquire_indices = np.flip(np.argsort(scores))[:n_acquire]
    xs = torch.Tensor(unlabeled_data[0].numpy()[acquire_indices])
    z_ids = torch.Tensor(unlabeled_data[1].numpy()[acquire_indices])

    # label the acquired data
    ys = data_label_fn(xs, z_ids)

    return xs, z_ids, ys


def active_train(latent_ensemble, dataloader, val_dataloader, train_fn, acquire_fn, logger, args):
    for tx in range(args.max_acquisitions):
        print('Acquisition step', tx)

        # save the current dataset
        logger.save_dataset(dataloader.dataset, tx)

        # reset and train the model on the current dataset
        if not args.use_latents: pass
        elif args.fitting: latent_ensemble.reset_latents()
        else: latent_ensemble.reset()

        hide_dims = [int(d) for d in args.hide_dims.split(',')] if args.hide_dims else []

        latent_ensemble, accs, latents = train_fn(dataloader,
                                                  val_dataloader,
                                                  latent_ensemble,
                                                  n_epochs=args.n_epochs,
                                                  freeze_ensemble=args.fitting,
                                                  return_logs=True,
                                                  hide_dims=hide_dims)

        # save the ensemble after training
        logger.save_ensemble(latent_ensemble, tx)

        # get new data using the BALD score
        new_data = acquire_fn(latent_ensemble)

        # save teh acquired data
        logger.save_acquisition_data(new_data, None, tx)#new_data, all_samples, tx)

        # add that data to the dataset
        dataloader.add(*new_data)


def run_active_throwing(args):
    objects = generate_objects(args.n_objects)
    hide_dims = [int(d) for d in args.hide_dims.split(',')] if args.hide_dims else []

    # use the sample_action function to get actions, and then preprocess to xs
    data_sampler_fn = lambda n_samples: generate_dataset(objects, args.n_samples, as_tensor=True, label=False)

    # forward pass of the latent ensemble, and marginalize the desired axes
    data_pred_fn = lambda latent_ensemble, unlabeled_data: get_predictions(latent_ensemble,
                                                                           unlabeled_data,
                                                                           n_latent_samples=10,
                                                                           marginalize_latents=not args.fitting,
                                                                           marginalize_ensemble=args.fitting,# and args.use_latents),
                                                                           hide_dims=hide_dims)

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
                                                            data_label_fn,
                                                            acquisition=args.acquisition.lower())

    # enable training with particle filter or with VI
    if args.fitting and args.use_particle_filter:
        train_fn = update_particle_filter
    else:
        train_fn = train


    print('Generating initialization and validation datasets')
    init_dataloader = ParallelDataLoader(TensorDataset(*generate_dataset(objects, 5*args.n_objects)),
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         n_dataloaders=args.n_models)
    # val_dataloader = ParallelDataLoader(TensorDataset(*generate_dataset(objects, 10*args.n_objects)),
    #                                  batch_size=args.batch_size,
    #                                  shuffle=True,
    #                                  n_dataloaders=1)
    val_dataloader = None


    # create a logger and save the object set (in vector form)
    logger = ActiveExperimentLogger.setup_experiment_directory(args)
    logger.save_objects(objects)

    # create the latent ensemble
    latent_ensemble = get_latent_ensemble(args)
    active_train(latent_ensemble, init_dataloader, val_dataloader, train_fn, acquire_fn, logger, args)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-acquisitions',
                        type=int,
                        default=125,
                        help='Number of iterations to run the main active learning loop for.')
    parser.add_argument('--exp-name', type=str, default='throwing', help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-models', type=int, default=10)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--n-acquire', type=int, default=10)
    parser.add_argument('--n-objects', type=int, default=10)
    parser.add_argument('--hide_dims', type=str, default='3')
    parser.add_argument('--acquisition', type=str, default='bald')

    parser.add_argument('--use-latents', action='store_true')
    parser.add_argument('--use-particle-filter', action='store_true')

    # The following arguments are used when we wanted to fit latents with an already trained model.
    parser.add_argument('--fitting', action='store_true', help='This will cause only the latents to update during training.')
    parser.add_argument('--latent-ensemble-exp-path', type=str, default='', help='Path to a trained latent ensemble.')
    parser.add_argument('--latent-ensemble-tx', type=int, default=-1, help='Timestep of the trained ensemble to evaluate.')

    return parser

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    run_active_throwing(args)

