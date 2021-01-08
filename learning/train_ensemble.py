import argparse
import numpy as np
import pickle

from torch.utils.data import DataLoader

from learning.active.active_train import active_train
from learning.active.train import train
from learning.domains.towers.active_utils import sample_unlabeled_data, get_predictions, get_labels, get_subset
from learning.domains.towers.tower_data import TowerDataset, TowerSampler
from learning.models.ensemble import Ensemble
from learning.models.gn import FCGN
from learning.active.utils import ActiveExperimentLogger


def train_ensemble(args):
    logger = ActiveExperimentLogger.setup_experiment_directory(args)
    
    # Initialize ensemble. 
    ensemble = Ensemble(base_model=FCGN,
                        base_args={'n_hidden': args.n_hidden, 'n_in': 14},
                        n_models=args.n_models)

    with open(args.data_fname, 'rb') as handle:
        dataset = pickle.load(handle)
    with open(args.data_fname, 'rb') as handle:
        val_dataset = pickle.load(handle)

    for k in dataset.tower_keys:
        shape = dataset.tower_tensors[k].shape
        dataset.tower_tensors[k][:, :, 7:9] += np.random.randn(shape[0]*shape[1]*2).reshape((shape[0], shape[1], 2))*0.0025*100
        val_dataset.tower_tensors[k][:, :, 7:9] += np.random.randn(shape[0]*shape[1]*2).reshape((shape[0], shape[1], 2))*0.0025*100

        train_mask = np.ones(dataset.tower_tensors[k].shape[0], dtype=bool)
        train_mask[::5] = False
        val_mask = ~train_mask

        dataset.tower_tensors[k] = dataset.tower_tensors[k][train_mask, ...]
        dataset.tower_labels[k] = dataset.tower_labels[k][train_mask, ...]

        val_dataset.tower_tensors[k] = val_dataset.tower_tensors[k][val_mask, ...]
        val_dataset.tower_labels[k] = val_dataset.tower_labels[k][val_mask, ...]

    dataset.get_indices()
    val_dataset.get_indices()

    sampler = TowerSampler(dataset=dataset,
                           batch_size=args.batch_size,
                           shuffle=True)
    dataloader = DataLoader(dataset,
                            batch_sampler=sampler)
    
    val_sampler = TowerSampler(dataset=val_dataset,
                               batch_size=args.batch_size,
                               shuffle=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_sampler=val_sampler)
    
    logger.save_dataset(dataset, 0)

    # Initialize and train models.
    ensemble.reset()
    for model in ensemble.models:
        train(dataloader, val_dataloader, model, args.n_epochs)
    
    logger.save_ensemble(ensemble, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-models', type=int, default=5, help='Number of models in the ensemble.')
    parser.add_argument('--n-hidden', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--data-fname', type=str, required=True)
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')
    args = parser.parse_args()

    train_ensemble(args)
