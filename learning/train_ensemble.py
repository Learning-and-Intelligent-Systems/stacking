import argparse
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

    sampler = TowerSampler(dataset=dataset,
                           batch_size=args.batch_size,
                           shuffle=True)
    dataloader = DataLoader(dataset,
                            batch_sampler=sampler)
    
    logger.save_dataset(dataset, 0)

    # Initialize and train models.
    ensemble.reset()
    for model in ensemble.models:
        train(dataloader, dataloader, model, args.n_epochs)
    
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
