import argparse
import pickle
import torch
from torch.utils.data import DataLoader

from learning.active.active_train import active_train
from learning.domains.towers.active_utils import sample_sequential_data, sample_unlabeled_data, get_predictions, get_labels, get_subset, PoolSampler, sample_next_block
from learning.domains.towers.tower_data import TowerDataset, TowerSampler, ParallelDataLoader
from learning.models.ensemble import Ensemble
from learning.models.bottomup_net import BottomUpNet
from learning.models.gn import FCGN, ConstructableFCGN, FCGNFC
from learning.models.lstm import TowerLSTM
from learning.train_latent import LatentEnsemble
from learning.active.utils import ActiveExperimentLogger
from agents.panda_agent import PandaAgent, PandaClientAgent
from tamp.misc import load_blocks


def run_active_towers(args):
    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # Initialize agent with supplied blocks (only works with args.block_set_fname set)
    if len(args.pool_fname) > 0:
        raise NotImplementedError()
    elif args.block_set_fname is not '':
        with open(args.block_set_fname, 'rb') as f:
            block_set = pickle.load(f)
    else:
        raise NotImplementedError()

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



    # Choose a sampler and check if we are limiting the blocks to work with.
    block_set = None
    if len(args.pool_fname) > 0:
        pool_sampler = PoolSampler(args.pool_fname)
        data_subset_fn = pool_sampler.get_subset
        data_sampler_fn = pool_sampler.sample_unlabeled_data
    elif args.block_set_fname is not '':
        data_subset_fn = get_subset
        with open(args.block_set_fname, 'rb') as f:
            # TODO: Unify block loading
            block_set = pickle.load(f)
            if args.exec_mode == "sim" or args.exec_mode == "real":
                block_set = load_blocks(fname=args.block_set_fname,
                                        num_blocks=10)
        data_sampler_fn = lambda n: sample_unlabeled_data(n, block_set=block_set)
    else:
        data_subset_fn = get_subset
        data_sampler_fn = sample_unlabeled_data

    # wrap the ensemble with latents
    if args.use_latents:
        ensemble = LatentEnsemble(ensemble, len(block_set), d_latents=4)

    # send to GPU if needed
    if torch.cuda.is_available():
        ensemble = ensemble.cuda()

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

        if args.sampler == 'sequential':
            data_sampler_fn = lambda n_samples: sample_sequential_data(block_set, dataset, n_samples)
        else:
            data_sampler_fn = lambda n: sample_unlabeled_data(n, block_set=block_set)
    elif args.sampler == 'sequential':
        print('Sampling initial dataset sequentially. Dataset NOT sampled on real robot.')
        towers_dict = sample_sequential_data(block_set, None, 40)
        towers_dict = get_labels(towers_dict, 'noisy-model', agent, logger, args.xy_noise)
        dataset = TowerDataset(towers_dict, augment=False, K_skip=1)

        val_towers_dict = sample_sequential_data(block_set, None, 40)
        val_towers_dict = get_labels(val_towers_dict, 'noisy-model', agent, logger, args.xy_noise)
        val_dataset = TowerDataset(val_towers_dict, augment=False, K_skip=1)

        if block_set is None:
            raise NotImplementedError()

        data_sampler_fn = lambda n_samples: sample_sequential_data(block_set, dataset, n_samples)
    else:
        print('Sampling initial dataset randomly.')
        towers_dict = sample_unlabeled_data(40, block_set=block_set)
        towers_dict = get_labels(towers_dict, args.exec_mode, agent, logger, args.xy_noise)
        dataset = TowerDataset(towers_dict, augment=True, K_skip=1)

        val_towers_dict = sample_unlabeled_data(40, block_set=block_set)
        val_towers_dict = get_labels(val_towers_dict, args.exec_mode, agent, logger, args.xy_noise)
        val_dataset = TowerDataset(val_towers_dict, augment=False, K_skip=1)


    if args.strategy == 'subtower-greedy':
        data_sampler_fn = lambda n_samples, bases: sample_next_block(n_samples, bases, block_set)
    if args.strategy == 'subtower':
        data_sampler_fn = lambda n: sample_unlabeled_data(n, block_set=block_set, range_n_blocks=(5, 5))

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

    # tell the "forward pass" of the latent ensemble to sample from the latents
    # and collapse the N_samples and N_models dimension into one
    if args.use_latents:
        data_pred_fn = lambda dataset, ensemble: get_predictions(
            dataset, ensemble, N_samples=20, use_latents=True)
    else:
        data_pred_fn = get_predictions

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
    parser.add_argument('--strategy', choices=['random', 'bald', 'subtower', 'subtower-greedy'], default='bald', help='[random] chooses towers randomly. [bald] scores each tower with the BALD score. [subtower-greedy] chooses a tower by adding blocks one at a time and keeping towers with the highest bald score [subtower] is similar to subtower-greedy, but we multiply the bald score of each tower by the probabiliy that the tower is constructible.')
    parser.add_argument('--sampler', choices=['random', 'sequential'], default='random', help='Choose how the unlabeled pool will be generated. Sequential assumes every tower has a stable base.')
    parser.add_argument('--pool-fname', type=str, default='')
    parser.add_argument('--model', default='fcgn', choices=['fcgn', 'fcgn-fc', 'fcgn-con', 'lstm', 'bottomup-shared', 'bottomup-unshared'])
    # simple-model: does not perturb the blocks, uses TowerPlanner to check constructability
    # noisy-model: perturbs the blocks, uses TowerPlanner to check constructability
    # sim: uses pyBullet with no noise
    # real: uses the real robot
    parser.add_argument('--exec-mode', default='noisy-model', choices=['simple-model', 'noisy-model', 'sim', 'real'])
    parser.add_argument('--xy-noise', default=0.003, type=float, help='Variance in the normally distributed noise in block placements (used when args.exec-mode==noisy-model)')
    parser.add_argument('--use-panda-server', action='store_true')
    parser.add_argument('--debug', action='store_true'),
    parser.add_argument('--use-latents', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    run_active_towers(args)
