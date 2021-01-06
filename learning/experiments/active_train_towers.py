import argparse
import pickle

from torch.utils.data import DataLoader

from learning.active.active_train import active_train
from learning.domains.towers.active_utils import sample_sequential_data, sample_unlabeled_data, get_predictions, get_labels, get_subset, PoolSampler
from learning.domains.towers.tower_data import TowerDataset, TowerSampler
from learning.models.ensemble import Ensemble
from learning.models.bottomup_net import BottomUpNet
from learning.models.gn import FCGN, ConstructableFCGN
from learning.models.lstm import TowerLSTM
from learning.active.utils import ActiveExperimentLogger


def run_active_towers(args):
    logger = ActiveExperimentLogger.setup_experiment_directory(args)
    
    # Initialize ensemble. 
    if args.model == 'fcgn':
        base_model = FCGN
        base_args = {'n_hidden': args.n_hidden, 'n_in': 14}
    elif args.model == 'fcgn-con':
        base_model = ConstructableFCGN
        base_args = {'n_hidden': args.n_hidden, 'n_in': 14}
    elif args.model == 'lstm':
        base_model = TowerLSTM
        base_args = {'n_hidden': args.n_hidden, 'n_in': 14}
    elif args.model == 'bottomup-shared':
        base_model = BottomUpNet
        base_args = {'n_hidden': args.n_hidden, 'n_in': 14, 'share_weights': True, 'max_blocks': 5}
    elif args.model == 'bottomup-unshared':
        base_model = BottomUpNet
        base_args = {'n_hidden': args.n_hidden, 'n_in': 14, 'share_weights': False, 'max_blocks': 5}

    else:
        raise NotImplementedError()

    ensemble = Ensemble(base_model=base_model,
                        base_args=base_args,
                        n_models=args.n_models)

    # Choose a sampler and check if we are limiting the blocks to work with.
    block_set = None
    if len(args.pool_fname) > 0:
        pool_sampler = PoolSampler(args.pool_fname)
        data_subset_fn = pool_sampler.get_subset
        data_sampler_fn = pool_sampler.sample_unlabeled_data
    elif args.block_set_fname is not '':
        data_subset_fn = get_subset
        with open(args.block_set_fname, 'rb') as f: block_set = pickle.load(f)
        data_sampler_fn = lambda n: sample_unlabeled_data(n, block_set=block_set)
    else:
        data_subset_fn = get_subset
        data_sampler_fn = sample_unlabeled_data

    # Sample initial dataset.
    if len(args.init_data_fname) > 0:
        # A good dataset to use is learning/data/random_blocks_(x40000)_5blocks_uniform_mass.pkl
        with open(args.init_data_fname, 'rb') as handle:
            towers_dict = pickle.load(handle)
        dataset = TowerDataset(towers_dict,
                               augment=True,
                               K_skip=4) # From this dataset, this means we start with 10 towers/size (before augmentation).
        with open('learning/data/random_blocks_(x1000.0)_constructable_val.pkl', 'rb') as handle:
            val_dict = pickle.load(handle)
        val_dataset = TowerDataset(val_dict, 
                                   augment=True,
                                   K_skip=10)
    
    else:
        towers_dict = sample_unlabeled_data(40, block_set=block_set)
        towers_dict = get_labels(towers_dict)
        dataset = TowerDataset(towers_dict, augment=True, K_skip=1)

        val_towers_dict = sample_unlabeled_data(40, block_set=block_set)
        val_towers_dict = get_labels(val_towers_dict)
        val_dataset = TowerDataset(val_towers_dict, augment=False, K_skip=1)

    if args.sampler == 'sequential':
        if block_set is None:
            raise NotImplementedError()
        data_sampler_fn = lambda n_samples: sample_sequential_data(block_set, dataset, n_samples)

    print(len(dataset), len(val_dataset)) 
    sampler = TowerSampler(dataset=dataset,
                           batch_size=args.batch_size,
                           shuffle=True,
                           oversample=False)
    dataloader = DataLoader(dataset,
                            batch_sampler=sampler)
    
    val_sampler = TowerSampler(dataset=val_dataset,
                               batch_size=args.batch_size,
                               shuffle=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_sampler=val_sampler)

    active_train(ensemble=ensemble, 
                 dataset=dataset, 
                 val_dataset=val_dataset,
                 dataloader=dataloader, 
                 val_dataloader=val_dataloader,
                 data_sampler_fn=data_sampler_fn, 
                 data_label_fn=get_labels, 
                 data_pred_fn=get_predictions,
                 data_subset_fn=data_subset_fn,
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
    parser.add_argument('--block-set-fname', type=str, default='')
    parser.add_argument('--n-train-init', type=int, default=100)
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--n-acquire', type=int, default=10)
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--strategy', choices=['random', 'bald'], default='bald')
    parser.add_argument('--sampler', choices=['random', 'sequential'], default='random', help='Choose how the unlabeled pool will be generated. Sequential assumes every tower has a stable base.')
    parser.add_argument('--pool-fname', type=str, default='')  
    parser.add_argument('--model', default='fcgn', choices=['fcgn', 'fcgn-con', 'lstm', 'bottomup-shared', 'bottomup-unshared'])      
    args = parser.parse_args()

    run_active_towers(args)
