import argparse
import pickle

from torch.utils.data import DataLoader

from learning.active.active_train import active_train
from learning.domains.towers.active_utils import sample_unlabeled_data, get_predictions, get_labels, get_subset
from learning.domains.towers.tower_data import TowerDataset, TowerSampler
from learning.models.ensemble import Ensemble
from learning.models.gn import FCGN
from learning.active.utils import ActiveExperimentLogger


def run_active_towers(args):
    logger = ActiveExperimentLogger.setup_experiment_directory(args)
    
    # Initialize ensemble. 
    ensemble = Ensemble(base_model=FCGN,
                        base_args={'n_hidden': args.n_hidden, 'n_in': 14},
                        n_models=args.n_models)

    # Sample initial dataset.
    if len(args.init_data_fname) > 0:
        # A good dataset to use is learning/data/random_blocks_(x40000)_5blocks_uniform_mass.pkl
        with open(args.init_data_fname, 'rb') as handle:
            towers_dict = pickle.load(handle)
        dataset = TowerDataset(towers_dict,
                               augment=True,
                               K_skip=400) # From this dataset, this means we start with 400 towers/size (before augmentation).
    else:
        # TODO: Generate random dataset.
        raise NotImplementedError()
    
    sampler = TowerSampler(dataset=dataset,
                           batch_size=args.batch_size,
                           shuffle=True)
    dataloader = DataLoader(dataset,
                            batch_sampler=sampler)
    
    # TODO: All these callback functions need to be rewritten for the towers dataset.
    active_train(ensemble=ensemble, 
                 dataset=dataset, 
                 dataloader=dataloader, 
                 data_sampler_fn=sample_unlabeled_data, 
                 data_label_fn=get_labels, 
                 data_pred_fn=get_predictions,
                 data_subset_fn=get_subset,
                 logger=logger, 
                 args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-acquisitions', 
                        type=int, 
                        default=1000,
                        help='Number of iterations to run the main active learning loop for.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-models', type=int, default=7, help='Number of models in the ensemble.')
    parser.add_argument('--n-hidden', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--init-data-fname', type=str, default='')
    parser.add_argument('--n-train-init', type=int, default=100)
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--n-acquire', type=int, default=10)
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--strategy', choices=['random', 'bald'], default='bald')    
    args = parser.parse_args()

    run_active_towers(args)
