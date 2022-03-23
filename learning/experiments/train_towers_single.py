import argparse
from pddlstream.language.statistics import DATA_DIR
import pickle
import torch
from torch.utils.data import DataLoader


from learning.domains.towers.tower_data import TowerDataset, ParallelDataLoader, TowerSampler
from learning.active.utils import ActiveExperimentLogger
from learning.active.train import train
from learning.models.gn import FCGN
from learning.models.ensemble import Ensemble
from learning.train_latent import train as train_latent
from learning.models.latent_ensemble import LatentEnsemble


def load_datasets_for_latent_ensemble(args):
    with open(args.train_dataset_fname, 'rb') as handle:
        train_towers_dict = pickle.load(handle)
    with open(args.val_dataset_fname, 'rb') as handle:
        val_towers_dict = pickle.load(handle)
    
    train_dataset = TowerDataset(train_towers_dict, augment=not args.disable_rotations)
    val_dataset = TowerDataset(val_towers_dict, augment=False)

    train_dataloader = ParallelDataLoader(dataset=train_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          n_dataloaders=args.n_models)
    val_dataloader = ParallelDataLoader(dataset=val_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        n_dataloaders=1) 

    return train_dataset, val_dataset, train_dataloader, val_dataloader

def load_datasets_for_ensemble(args):
    with open(args.train_dataset_fname, 'rb') as handle:
        train_towers_dict = pickle.load(handle)
    with open(args.val_dataset_fname, 'rb') as handle:
        val_towers_dict = pickle.load(handle)

    # for tx in range(train_towers_dict['2block']['towers'].shape[0]):
    #     print(train_towers_dict['2block']['towers'][tx, :, 10:14])

    train_dataset = TowerDataset(train_towers_dict, augment=False)
    val_dataset = TowerDataset(val_towers_dict, augment=False)
    sampler = TowerSampler(dataset=train_dataset,
                           batch_size=args.batch_size,
                           shuffle=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=sampler)

    val_sampler = TowerSampler(dataset=val_dataset,
                               batch_size=args.batch_size,
                               shuffle=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_sampler=val_sampler)
    return train_dataset, val_dataset, train_dataloader, val_dataloader


def initialize_model(args):
    n_in = 13
    if args.com_repr == 'removed':
        n_in = 10
    if args.model == 'fcgn':
        base_model = FCGN
        base_args = {'n_hidden': args.n_hidden, 'n_in': n_in, 'remove_com': args.com_repr == 'removed' }
    else:
        raise NotImplementedError()

    ensemble = Ensemble(base_model=base_model,
                        base_args=base_args,
                        n_models=args.n_models)

    # wrap the ensemble with latents
    if args.use_latents:
        ensemble = LatentEnsemble(ensemble, args.n_blocks, d_latents=3)

    return ensemble


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')

    # Dataset parameters. 
    parser.add_argument('--train-dataset-fname', type=str, required=True)
    parser.add_argument('--val-dataset-fname', type=str, required=True)
    parser.add_argument('--n-blocks', type=int, required=True)
    parser.add_argument('--block-set-fname', type=str, required=True)  # Not needed for training but will be needed for fitting.
    # Model parameters.
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-models', type=int, default=7, help='Number of models in the ensemble.')
    parser.add_argument('--n-hidden', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--model', default='fcgn', choices=['fcgn', 'fcgn-fc', 'fcgn-con', 'lstm', 'bottomup-shared', 'bottomup-unshared'])
    parser.add_argument('--com-repr', type=str, choices=['latent', 'explicit', 'removed'], required=True,
                        help='Explicit specifies the true CoM for each block. Latent has a LV for the CoM. Removed completely removes any CoM repr.')
    parser.add_argument('--disable-rotations', action='store_true', default=False)
    args = parser.parse_args()
    args.use_latents = args.com_repr == 'latent'
    args.fit = False
    args.max_acquisitions = 1

    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # Build model.
    ensemble = initialize_model(args)
    if torch.cuda.is_available():
        ensemble = ensemble.cuda()

    # Train model.
    if args.use_latents:
        # Load training and val datasets.
        train_dataset, val_dataset, train_dataloader, val_dataloader = load_datasets_for_latent_ensemble(args)
        train_latent(dataloader=train_dataloader,
                     val_dataloader=val_dataloader,
                     latent_ensemble=ensemble,
                     n_epochs=args.n_epochs,
                     freeze_ensemble=False,
                     args=args,
                     show_epochs=True)
    else:
        train_dataset, val_dataset, train_dataloader, val_dataloader = load_datasets_for_ensemble(args)
        ensemble.reset()
        for model in ensemble.models:
            train(train_dataloader, val_dataloader, model, args.n_epochs)


    # Save model.
    logger.save_dataset(dataset=train_dataset, tx=0)
    logger.save_val_dataset(val_dataset=val_dataset, tx=0)
    logger.save_ensemble(ensemble=ensemble, tx=0)

