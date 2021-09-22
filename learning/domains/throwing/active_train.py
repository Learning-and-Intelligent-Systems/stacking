import argparse
import numpy as np
import torch
import copy
from torch.utils.data import TensorDataset, DataLoader
import pickle

from learning.active.acquire import bald_diagonal_gaussian
from learning.active.utils import ActiveExperimentLogger
from learning.domains.throwing.particle_filter import update_particle_filter
from learning.domains.throwing.throwing_data import generate_objects, generate_dataset, label_actions, ParallelDataLoader, xs_to_actions, generate_dataset_with_repeated_actions
from learning.domains.throwing.train_latent import get_predictions, train
from learning.models.ensemble import Ensemble
from learning.models.mlp import FeedForward, FeedForwardWithSkipConnections
from learning.models.latent_ensemble import ThrowingLatentEnsemble, convert_to_particle_filter_latent_ensemble

def get_latent_ensemble(args):
    n_latents = args.n_objects
    d_observe = 12
    d_latents = 2
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
        ensemble = Ensemble(base_model=FeedForwardWithSkipConnections,
                            base_args={
                                        'd_in': d_observe + d_latents - len(hide_dims),
                                        'd_out': d_pred,
                                        'd_latent': d_latents,
                                        'h_dims': [16, 64]#[64, 32, 32]
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
                       acquisition='bald',
                       per_object=False):

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
    if per_object:
        all_xs = unlabeled_data[0].numpy()
        all_zs = unlabeled_data[1].numpy()
        n_objects = len(set(unlabeled_data[1].numpy().tolist())) 

        xs_list, z_ids_list = [], []
        for ix in range(n_objects):
            obj_ix_xs = all_xs[all_zs == ix]
            obj_ix_scores = scores[all_zs == ix]
            obj_ix_zs = all_zs[all_zs == ix]

            acquire_indices = np.flip(np.argsort(obj_ix_scores))[:1]
            xs = torch.Tensor(obj_ix_xs[acquire_indices])
            z_ids = torch.Tensor(obj_ix_zs[acquire_indices])
            
            xs_list.append(xs)
            z_ids_list.append(z_ids)
        
        xs = torch.cat(xs_list, dim=0)
        z_ids = torch.cat(z_ids_list, dim=0)
    else:
        acquire_indices = np.flip(np.argsort(scores))[:n_acquire]
        xs = torch.Tensor(unlabeled_data[0].numpy()[acquire_indices])
        z_ids = torch.Tensor(unlabeled_data[1].numpy()[acquire_indices])

    # label the acquired data
    ys = data_label_fn(xs, z_ids)

    return (xs, z_ids, ys), unlabeled_data


def active_train(latent_ensemble, dataloader, val_dataloader, train_fn, acquire_fn, logger, args):
    for tx in range(args.max_acquisitions):
        print('Acquisition step', tx)

        # save the current dataset
        logger.save_dataset(dataloader.dataset, tx)
        logger.save_val_dataset(val_dataloader.dataset, tx)

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
                                                  hide_dims=hide_dims,
                                                  use_normalization=args.use_normalization)

        # save the ensemble after training
        logger.save_ensemble(latent_ensemble, tx)

        # get new data using the BALD score
        new_data, unlabeled_data = acquire_fn(latent_ensemble)

        # save teh acquired data
        logger.save_acquisition_data(new_data, unlabeled_data, tx)#new_data, all_samples, tx)

        # add that data to the dataset
        train_data, val_data = split_data(new_data, 2)
        dataloader.add(*train_data)
        val_dataloader.add(*val_data)


def split_data(data, n_val):
    """
    Choose n_val of the chosen data points to add to the validation set.
    Return 2 tower_dict structures.
    """    
    total = data[0].shape[0]
    val_ixs = np.random.choice(np.arange(0, total), n_val, replace=False)
    
    train_mask = np.ones(total, dtype=bool)
    train_mask[val_ixs] = False
    train_data = (data[0][train_mask, ...].clone(), data[1][train_mask].clone(), data[2][train_mask].clone())

    val_mask = np.zeros(total, dtype=bool)
    val_mask[val_ixs] = True
    val_data = (data[0][val_mask, ...].clone(), data[1][val_mask].clone(), data[2][val_mask].clone())

    return train_data, val_data


def run_active_throwing(args):
    if len(args.object_fname) == 0:
        objects = generate_objects(args.n_objects)
    else:
        with open(args.object_fname, 'rb') as handle:
            objects = pickle.load(handle)
    
    hide_dims = [int(d) for d in args.hide_dims.split(',')] if args.hide_dims else []

    # use the sample_action function to get actions, and then preprocess to xs
    data_sampler_fn = lambda n_samples: generate_dataset(objects, args.n_samples, as_tensor=True, label=False)

    # forward pass of the latent ensemble, and marginalize the desired axes
    data_pred_fn = lambda latent_ensemble, unlabeled_data: get_predictions(latent_ensemble,
                                                                           unlabeled_data,
                                                                           n_latent_samples=args.n_latent_samples,
                                                                           marginalize_latents=not args.fitting,
                                                                           marginalize_ensemble=args.fitting,# and args.use_latents),
                                                                           hide_dims=hide_dims,
                                                                           use_normalization=args.use_normalization,
                                                                           return_normalized=True) # we want to compute BALD in normalized space

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
                                                            acquisition=args.acquisition.lower(),
                                                            per_object=False)

    # enable training with particle filter or with VI
    if args.fitting and args.use_particle_filter:
        train_fn = update_particle_filter
    else:
        train_fn = train


    print('Generating initialization and validation datasets')
    init_dataloader = ParallelDataLoader(TensorDataset(*generate_dataset(objects, 5*args.n_objects, duplicate=False)),
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         n_dataloaders=args.n_models)
    val_dataloader = ParallelDataLoader(TensorDataset(*generate_dataset(objects, 5*args.n_objects)),
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     n_dataloaders=1)


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
    parser.add_argument('--n-latent-samples', type=int, default=10)
    parser.add_argument('--n-epochs', type=int, default=500)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--n-acquire', type=int, default=10)
    parser.add_argument('--n-objects', type=int, default=10)
    parser.add_argument('--hide_dims', type=str, default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--acquisition', type=str, default='bald')
    parser.add_argument('--object-fname', type=str, default='')

    parser.add_argument('--use-latents', action='store_true')
    parser.add_argument('--use-normalization', action='store_true')
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

