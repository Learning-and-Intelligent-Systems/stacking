import argparse
from pddlstream.language.statistics import DATA_DIR
import pickle

from learning.domains.towers.tower_data import TowerDataset, ParallelDataLoader
from learning.active.utils import ActiveExperimentLogger
from learning.models.gn import FCGN
from learning.models.ensemble import Ensemble
from learning.train_latent import train as train_latent
from learning.models.latent_ensemble import LatentEnsemble


def load_datasets(args):
    with open(args.train_dataset_fname, 'rb') as handle:
        train_towers_dict = pickle.load(handle)
    with open(args.val_dataset_fname, 'rb') as handle:
        val_towers_dict = pickle.load(handle)
    
    train_dataset = TowerDataset(train_towers_dict, augment=True)
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


def initialize_model(args):
    n_in = 14
    if args.model == 'fcgn':
        base_model = FCGN
        base_args = {'n_hidden': args.n_hidden, 'n_in': n_in, 'remove_com': False }
    else:
        raise NotImplementedError()

    ensemble = Ensemble(base_model=base_model,
                        base_args=base_args,
                        n_models=args.n_models)

    # wrap the ensemble with latents
    if args.use_latents:
        ensemble = LatentEnsemble(ensemble, args.n_blocks, d_latents=4)

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
    args = parser.parse_args()
    args.use_latents = True
    args.fit = False

    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # Load training and val datasets.
    train_dataset, val_dataset, train_dataloader, val_dataloader = load_datasets(args)

    # Build model.
    ensemble = initialize_model(args)

    # Train model.
    train_latent(dataloader=train_dataloader,
                 val_dataloader=val_dataloader,
                 latent_ensemble=ensemble,
                 n_epochs=args.n_epochs,
                 freeze_ensemble=False,
                 args=args,
                 show_epochs=True)

    # Save model.
    logger.save_dataset(dataset=train_dataset, tx=0)
    logger.save_val_dataset(val_dataset=val_dataset, tx=0)
    logger.save_ensemble(ensemble=ensemble, tx=0)

