import argparse
import pickle
import os
import numpy as np

from torch.utils.data import DataLoader

from learning.active.active_train import active_train
from learning.domains.towers.active_utils import sample_sequential_data, sample_unlabeled_data, \
            get_sequential_predictions, get_predictions, get_labels, get_subset, PoolSampler
from learning.domains.towers.tower_data import TowerDataset, TowerSampler
from learning.models.ensemble import Ensemble
from learning.models.bottomup_net import BottomUpNet
from learning.models.gn import FCGN, ConstructableFCGN, FCGNFC
from learning.models.lstm import TowerLSTM
from learning.active.utils import ActiveExperimentLogger
from learning.active.train import train
from learning.active.acquire import choose_acquisition_data
from learning.active.active_train import split_data
from agents.panda_agent import PandaAgent, PandaClientAgent
from block_utils import block_conflicts
from tamp.misc import load_blocks



def tower_index(towers_data, tower):
    for tdi, (tdi_tower, _, _) in enumerate(towers_data):
        if np.array_equal(tower, tdi_tower):
            return tdi
    return None
    

def recover_labels(logger, args, agent):
    towers_data = logger.get_towers_data(logger.acquisition_step)
    acquisition_data = logger.get_unlabeled_acquisition_data()
    print(f"Already found {len(towers_data)} towers in acquisition step {logger.acquisition_step}.")
    for k, data_k in acquisition_data.items():
        towers_k = data_k['towers']
        block_ids_k = data_k['block_ids']
        labels_k = np.zeros(towers_k.shape[0])
        for ti in range(towers_k.shape[0]):
            ix = tower_index(towers_data, towers_k[ti])
            if ix is not None:
                labels_k[ti] = towers_data[ix][2] # towers_data is a list of [tower, block_ids, label] lists
            else:
                tower_to_label = {k : {'towers': np.array([towers_k[ti]]), \
                                        'block_ids': np.array([block_ids_k[ti]])}}
                labeled_tower = get_labels(tower_to_label, args.exec_mode, agent, logger, args.xy_noise, save_tower=True)
                labels_k[ti] = labeled_tower[k]['labels'][0]
        acquisition_data[k]['labels'] = labels_k
    return acquisition_data
    

def initialize_ensemble(args):
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
    return ensemble
    
def setup_active_train(dataset,
                        val_dataset,
                        dataloader,
                        val_dataloader,
                        logger, 
                        data_sampler_fn, 
                        data_label_fn, 
                        data_pred_fn, 
                        data_subset_fn, 
                        agent, 
                        args):
    ensemble = initialize_ensemble(args)
    
    # training stopped after data was acquired but before adding it to dataset
    print(f"Resuming from Logger acquisition step: {logger.acquisition_step}, tower step {logger.tower_counter}")
    acquired_data, pool_data = logger.load_acquisition_data(logger.acquisition_step)
    next_dataset = logger.load_dataset(logger.acquisition_step+1)
    
    if acquired_data and not next_dataset:
        train_data, val_data = split_data(acquired_data, n_val=2)
        dataset.add_to_dataset(train_data)
        val_dataset.add_to_dataset(val_data)
        logger.acquisition_step += 1

    else:
        # training stopped after dataset was saved but before model was trained and saved
        ensemble = logger.get_ensemble(logger.acquisition_step).cuda()
        if dataset and not ensemble:
            ensemble = initialize_ensemble(args)
            ensemble.reset()
            for model in ensemble.models:
                train(dataloader, val_dataloader, model, args.n_epochs)
            logger.save_ensemble(ensemble, logger.acquisition_step)
        
        # training stopped after model was trained but before (or during) data acquisition
        # NOTE: if training stops between getting unlabeled samples and starting to label them
        # then this won't work (but I think that scenario is highly unlikely)
        acquired_data, _ = logger.load_acquisition_data(logger.acquisition_step)
        if ensemble and not acquired_data:
            acquisition_data = recover_labels(logger, args, agent)
            logger.save_acquisition_data(acquisition_data, None, logger.acquisition_step)

            # Add to dataset.
            train_data, val_data = split_data(acquisition_data, n_val=2)
            dataset.add_to_dataset(train_data)
            val_dataset.add_to_dataset(val_data)
        
    return ensemble

def restart_active_towers(exp_path, args):
    logger = ActiveExperimentLogger.get_experiments_logger(exp_path, args)
    
    # starting dataset (must be at least one in the exp_path)
    dataset = logger.load_dataset(logger.acquisition_step)
    val_dataset = logger.load_val_dataset(logger.acquisition_step)
    
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
    
    # only works with args.block_set_fname set
    if args.block_set_fname is '':
        raise NotImplementedError() 
    
    if args.exec_mode == 'simple-model' or args.exec_mode == 'noisy-model':
        agent = None
    elif args.exec_mode == 'sim' or args.exec_mode == 'real':
        if args.use_panda_server:
            agent = PandaClientAgent()
        else:
            agent = PandaAgent(block_set)

    # Choose a sampler and check if we are limiting the blocks to work with.
    block_set = None
    if len(args.pool_fname) > 0:
        pool_sampler = PoolSampler(args.pool_fname)
        data_subset_fn = pool_sampler.get_subset
        data_sampler_fn = pool_sampler.sample_unlabeled_data
    elif args.block_set_fname is not '':
        data_subset_fn = get_subset
        with open(args.block_set_fname, 'rb') as f: 
            block_set = pickle.load(f)
            if args.exec_mode == "sim" or args.exec_mode == "real":
                block_set = load_blocks(fname=args.block_set_fname,
                                        num_blocks=10)
        data_sampler_fn = lambda n: sample_unlabeled_data(n, block_set=block_set)
    else:
        data_subset_fn = get_subset
        data_sampler_fn = sample_unlabeled_data

    if args.sampler == 'sequential':
        data_sampler_fn = lambda n_samples: sample_sequential_data(block_set, dataset, n_samples)

    print("Setting up dataset")
    ensemble = setup_active_train(dataset,
                                    val_dataset,
                                    dataloader=dataloader,
                                    val_dataloader=val_dataloader,
                                    logger=logger, 
                                    data_sampler_fn=data_sampler_fn, 
                                    data_label_fn=get_labels, 
                                    data_pred_fn=get_predictions, 
                                    data_subset_fn=data_subset_fn, 
                                    agent=agent, 
                                    args=args)

    print("Restarting active learning")
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
                 agent=agent,
                 args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, default='', help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--debug', action='store_true')
    
    # NOTE: only use the below arguments if you want to overwrite the arguments used
    # in the initial run!
    parser.add_argument('--max-acquisitions', type=int,
                        help='Number of iterations to run the main active learning loop for.')
    parser.add_argument('--block-set-fname', type=str, help='File containing a list of AT LEAST 5 blocks (block_utils.Object) where the block.name is formatted obj_#')
    parser.add_argument('--xy-noise', type=float, help='Variance in the normally distributed noise in block placements (used when args.exec-mode==noisy-model)')
    restart_args = parser.parse_args()

    if restart_args.debug:
        import pdb; pdb.set_trace()

    with open(os.path.join(restart_args.exp_path, 'args.pkl'), 'rb') as f: 
        args = pickle.load(f)

    # replace args (if set in restart_args)
    if restart_args.max_acquisitions:
        args.max_acquisitions = restart_args.max_acquisitions
    if restart_args.block_set_fname:
        with open(args.block_set_fname, 'rb') as f: 
            old_block_set = pickle.load(f)
        with open(args.block_set_fname, 'rb') as f: 
            new_block_set = pickle.load(f)
        assert (not block_conflicts(old_block_set+new_block_set)), \
                'There are conflicting block names in the previously used and new block sets'
        args.block_set_fname = restart_args.block_set_fname
    if restart_args.xy_noise:
        args.xy_noise = restart_args.xy_noise
    
    restart_active_towers(restart_args.exp_path, args)
