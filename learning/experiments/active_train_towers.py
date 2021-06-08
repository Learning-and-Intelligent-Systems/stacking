import argparse
import pickle
import torch
from torch.utils.data import DataLoader

from learning.active.active_train import active_train
from learning.domains.towers.active_utils import sample_sequential_data, sample_unlabeled_data, get_predictions, get_labels, get_subset, sample_next_block
from learning.domains.towers.tower_data import TowerDataset, TowerSampler, ParallelDataLoader
from learning.models.ensemble import Ensemble
from learning.models.latent_ensemble import LatentEnsemble
from learning.models.bottomup_net import BottomUpNet
from learning.models.gn import FCGN, ConstructableFCGN, FCGNFC
from learning.models.lstm import TowerLSTM
from learning.train_latent import LatentEnsemble
from learning.active.utils import ActiveExperimentLogger
from agents.panda_agent import PandaAgent, PandaClientAgent
from tamp.misc import load_blocks

def initialize_model(args, n_blocks):
    if args.model == 'fcgn':
        base_model = FCGN
        base_args = {'n_hidden': args.n_hidden, 'n_in': 14}
    elif args.model == 'fcgn-fc':
        base_model = FCGNFC
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

    # wrap the ensemble with latents
    if args.use_latents:
        ensemble = LatentEnsemble(ensemble, n_blocks, d_latents=4)

    return ensemble


def load_latent_ensemble(args):
    assert(args.use_latents and len(args.latent_ensemble_exp_path) > 0 and args.latent_ensemble_tx >= 0)
    logger = ActiveExperimentLogger.get_experiments_logger(args.latent_ensemble_exp_path, args)
    ensemble = logger.get_ensemble(args.latent_ensemble_tx)
    return ensemble


def get_initial_dataset(args, block_set, agent, logger):
    # If we are fitting the latents, start with an empty dataset.
    if args.fit_latents:
        towers_dict = sample_sequential_data(block_set, None, 0)
        towers_dict = get_labels(towers_dict, 'noisy-model', agent, logger, args.xy_noise)
        dataset = TowerDataset(towers_dict, augment=True, K_skip=1)

        val_towers_dict = sample_sequential_data(block_set, None, 0)
        val_towers_dict = get_labels(val_towers_dict, 'noisy-model', agent, logger, args.xy_noise)
        val_dataset = TowerDataset(val_towers_dict, augment=False, K_skip=1)    
        return dataset, val_dataset

    # Sample initial dataset. 
    if len(args.init_data_fname) > 0:
        print(f'Loading an initial dataset from {args.init_data_fname}')
        # A good dataset to use is learning/data/random_blocks_(x40000)_5blocks_uniform_mass.pkl
        with open(args.init_data_fname, 'rb') as handle:
            towers_dict = pickle.load(handle)
        dataset = TowerDataset(towers_dict, augment=True) # From this dataset, this means we start with 10 towers/size (before augmentation).
        with open(args.val_data_fname, 'rb') as handle:
            val_dict = pickle.load(handle)
        val_dataset = TowerDataset(val_dict, augment=False)
    else:
        # If an initial dataset isn't given, sample one based on the sampler type.
        if args.sampler == 'sequential':
            print('Sampling initial dataset sequentially. Dataset NOT sampled on real robot.')
            towers_dict = sample_sequential_data(block_set, None, 40)
            towers_dict = get_labels(towers_dict, 'noisy-model', agent, logger, args.xy_noise)
            dataset = TowerDataset(towers_dict, augment=False, K_skip=1)

            val_towers_dict = sample_sequential_data(block_set, None, 40)
            val_towers_dict = get_labels(val_towers_dict, 'noisy-model', agent, logger, args.xy_noise)
            val_dataset = TowerDataset(val_towers_dict, augment=False, K_skip=1)            
        else:
            print('Sampling initial dataset randomly.')
            towers_dict = sample_unlabeled_data(40, block_set=block_set)
            towers_dict = get_labels(towers_dict, args.exec_mode, agent, logger, args.xy_noise)
            dataset = TowerDataset(towers_dict, augment=True, K_skip=1)

            val_towers_dict = sample_unlabeled_data(40, block_set=block_set)
            val_towers_dict = get_labels(val_towers_dict, args.exec_mode, agent, logger, args.xy_noise)
            val_dataset = TowerDataset(val_towers_dict, augment=False, K_skip=1)
    
    return dataset, val_dataset


def get_dataloaders(args, dataset, val_dataset):
    #print(len(dataset), len(val_dataset))
    if args.use_latents:
        dataloader = ParallelDataLoader(dataset,
            batch_size=args.batch_size, shuffle=True, n_dataloaders=args.n_models)
        val_dataloader = ParallelDataLoader(val_dataset,
            batch_size=args.batch_size, shuffle=False, n_dataloaders=1)

    else:
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
    return dataloader, val_dataloader


def get_sampler_fn(args, block_set):
    if args.n_samples < 100000:
        print('[WARNING] Running with fewer than 100k unlabeled samples.')
    # Certain strategies require specific data-sampler functions.
    if args.strategy == 'subtower-greedy':
        data_sampler_fn = lambda n_samples, bases: sample_next_block(n_samples, bases, block_set)
    elif args.strategy == 'subtower':
        data_sampler_fn = lambda n: sample_unlabeled_data(n, block_set=block_set, range_n_blocks=(5, 5))
    else:
        # Otherwise we defaul to whatever --sampler was directly specified.
        if args.sampler == 'sequential':
            data_sampler_fn = lambda n_samples: sample_sequential_data(block_set, dataset, n_samples)
        else:
            data_sampler_fn = lambda n: sample_unlabeled_data(n, block_set=block_set)
    return data_sampler_fn


def run_active_towers(args):
    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # Initialize agent with supplied blocks (only works with args.block_set_fname set)
    if args.block_set_fname is not '':
        with open(args.block_set_fname, 'rb') as f:
            block_set = pickle.load(f)
    else:
        raise NotImplementedError()

    # Set the agent used to get labels/perform experiments.
    if args.exec_mode == 'simple-model' or args.exec_mode == 'noisy-model':
        agent = None
    elif args.exec_mode == 'sim' or args.exec_mode == 'real':
        if args.use_panda_server:
            agent = PandaClientAgent()
        else:
            block_set = load_blocks(fname=args.block_set_fname,
                                    num_blocks=10)
            agent = PandaAgent(block_set)

    # Initialize ensemble.
    if args.fit_latents:
        ensemble = load_latent_ensemble(args)
    else:
        ensemble = initialize_model(args, len(block_set))    
    if torch.cuda.is_available():
        ensemble = ensemble.cuda()

    # Get an initial dataset.
    dataset, val_dataset = get_initial_dataset(args, block_set, agent, logger)
    dataloader, val_dataloader = get_dataloaders(args, dataset, val_dataset)
    if args.fit_latents:
        val_dataset, val_dataloader = None, None

    # Get active learning helper functions.
    data_subset_fn = get_subset
    data_sampler_fn = get_sampler_fn(args, block_set)
    # tell the "forward pass" of the latent ensemble to sample from the latents
    # and collapse the N_samples and N_models dimension into one
    if args.use_latents:
        if args.sample_joint:
            collapse_ensemble = False
            collapse_latents = False
        elif args.fit_latents:
            collapse_ensemble = True
            collapse_latents = False
        else:
            collapse_ensemble = False
            collapse_latents = True

        data_pred_fn = lambda dataset, ensemble: get_predictions(
            dataset, ensemble, N_samples=10, use_latents=True,
            collapse_latents=collapse_latents, collapse_ensemble=collapse_ensemble)
    else:
        data_pred_fn = get_predictions    

    # Start training.
    print('Starting training from scratch.')
    if args.exec_mode == 'real':
        input('Press enter to confirm you want to start training from scratch.')
    active_train(ensemble=ensemble,
                 dataset=dataset,
                 val_dataset=val_dataset,
                 dataloader=dataloader,
                 val_dataloader=val_dataloader,
                 data_sampler_fn=data_sampler_fn,
                 data_label_fn=get_labels,
                 data_pred_fn=data_pred_fn,
                 data_subset_fn=data_subset_fn,
                 logger=logger,
                 agent=agent,
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
    parser.add_argument('--val-data-fname', type=str, default='')
    parser.add_argument('--block-set-fname', type=str, default='', help='File containing a list of AT LEAST 5 blocks (block_utils.Object) where the block.name is formatted obj_#')
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--n-acquire', type=int, default=10)
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--sample-joint', action='store_true', help='Whether BALD should sample from the joint distribution or marginals (default).')
    parser.add_argument('--strategy', choices=['random', 'bald', 'subtower', 'subtower-greedy'], default='bald', help='[random] chooses towers randomly. [bald] scores each tower with the BALD score. [subtower-greedy] chooses a tower by adding blocks one at a time and keeping towers with the highest bald score [subtower] is similar to subtower-greedy, but we multiply the bald score of each tower by the probabiliy that the tower is constructible.')
    parser.add_argument('--sampler', choices=['random', 'sequential'], default='random', help='Choose how the unlabeled pool will be generated. Sequential assumes every tower has a stable base.')
    parser.add_argument('--model', default='fcgn', choices=['fcgn', 'fcgn-fc', 'fcgn-con', 'lstm', 'bottomup-shared', 'bottomup-unshared'])
    # simple-model: does not perturb the blocks, uses TowerPlanner to check constructability
    # noisy-model: perturbs the blocks, uses TowerPlanner to check constructability
    # sim: uses pyBullet with no noise
    # real: uses the real robot
    parser.add_argument('--exec-mode', default='noisy-model', choices=['simple-model', 'noisy-model', 'sim', 'real'])
    parser.add_argument('--xy-noise', default=0.003, type=float, help='Variance in the normally distributed noise in block placements (used when args.exec-mode==noisy-model)')
    parser.add_argument('--use-panda-server', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use-latents', action='store_true')
    # The following arguments are used when we wanted to fit latents with an already trained model.
    parser.add_argument('--fit-latents', action='store_true', help='This will cause only the latents to update during training.')
    parser.add_argument('--latent-ensemble-exp-path', type=str, default='', help='Path to a trained latent ensemble.')
    parser.add_argument('--latent-ensemble-tx', type=int, default=-1, help='Timestep of the trained ensemble to evaluate.')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    run_active_towers(args)
