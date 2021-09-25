import re
import numpy as np
import os
import pickle
import time
import torch
import datetime

from torch.utils.data import DataLoader

from learning.models.ensemble import Ensemble
from learning.models.mlp_dropout import MLP
from learning.models.latent_ensemble import LatentEnsemble


class ExperimentLogger:

    def __init__(self, exp_path):
        self.exp_path = exp_path

        with open(os.path.join(self.exp_path, 'args.pkl'), 'rb') as handle:
            self.args = pickle.load(handle)

    @staticmethod
    def setup_experiment_directory(args):
        """
        Setup the directory structure to store models, figures, datasets
        and parameters relating to an experiment.
        """
        root = 'learning/active/experiments'
        if not os.path.exists(root): os.makedirs(root)

        exp_name = args.exp_name if len(args.exp_name) > 0 else 'exp'
        ts = time.strftime('%Y%m%d-%H%M%S')
        exp_dir = '%s-%s' % (exp_name, ts)
        exp_path = os.path.join(root, exp_dir)

        os.mkdir(exp_path)
        os.mkdir(os.path.join(exp_path, 'figures'))
        os.mkdir(os.path.join(exp_path, 'models'))
        os.mkdir(os.path.join(exp_path, 'datasets'))

        with open(os.path.join(exp_path, 'args.pkl'), 'wb') as handle:
            pickle.dump(args, handle)

        return ExperimentLogger(exp_path)

    def save_dataset(self, dataset, fname):
        with open(os.path.join(self.exp_path, 'datasets', fname), 'wb') as handle:
            pickle.dump(dataset, handle)

    def load_dataset(self, fname):
        with open(os.path.join(self.exp_path, 'datasets', fname), 'rb') as handle:
            dataset = pickle.load(handle)
        return dataset

    def get_figure_path(self, fname):
        return os.path.join(self.exp_path, 'figures', fname)

    def save_model(self, model, fname):
        torch.save(model.state_dict(), os.path.join(self.exp_path, 'models', fname))

    def load_model(self, fname):
        model = MLP(n_hidden=self.args.n_hidden, dropout=self.args.dropout)
        model.load_state_dict(torch.load(os.path.join(self.exp_path, 'models', fname)))
        return model

    def get_ensemble(self):
        ensemble = []
        for mx in range(0, self.args.n_models):
            ensemble.append(self.load_model('net_%d.pt' % mx))
        return ensemble

class ActiveExperimentLogger:

    def __init__(self, exp_path, use_latents=False):
        self.exp_path = exp_path
        self.acquisition_step = 0
        self.tower_counter = 0
        self.use_latents = use_latents

        with open(os.path.join(self.exp_path, 'args.pkl'), 'rb') as handle:
            self.args = pickle.load(handle)

        self.update_max_acquisitions()

    def update_max_acquisitions(self):
        """ Check that self.args.max_acquisitions corresponds to the number of
        acquisition steps actually performed.
        """
        aq_files = os.listdir(os.path.join(self.exp_path, 'acquisition_data'))
        max_aq = np.max([int(s.split('_')[1].split('.')[0]) for s in aq_files]) + 1
        if max_aq != self.args.max_acquisitions:
            print('[WARNING] Only %d acquisition steps have been executed so far.' % max_aq)
            self.args.max_acquisitions = max_aq
    
    def get_acquisition_params(self):
        """ Return parameters useful for knowing how much data was collected.
        :return: init_dataset_size, n_acquire
        """
        init_dataset = self.load_dataset(0)
        return len(init_dataset)//4, self.args.n_acquire

    @staticmethod
    def get_experiments_logger(exp_path, args):
        logger = ActiveExperimentLogger(exp_path, use_latents=args.use_latents)
        dataset_path = os.path.join(exp_path, 'datasets')
        dataset_files = os.listdir(dataset_path)
        if len(dataset_files) == 0:
            raise Exception('No datasets found on args.exp_path. Cannot restart active training.')
        txs = []
        for file in dataset_files:
            matches = re.match(r'active_(.*).pkl', file)
            if matches: # sometimes system files are saved here, don't parse these
                txs += [int(matches.group(1))]
        logger.acquisition_step = max(txs)
        
        # save potentially new args
        with open(os.path.join(exp_path, 'args_restart.pkl'), 'wb') as handle:
            pickle.dump(args, handle)

        return logger

    @staticmethod
    def setup_experiment_directory(args):
        """
        Setup the directory structure to store models, figures, datasets
        and parameters relating to an experiment.
        """
        root = 'learning/experiments/logs'
        if not os.path.exists(root): os.makedirs(root)

        exp_name = args.exp_name if len(args.exp_name) > 0 else 'exp'
        ts = time.strftime('%Y%m%d-%H%M%S')
        exp_dir = '%s-%s' % (exp_name, ts)
        exp_path = os.path.join(root, exp_dir)

        os.mkdir(exp_path)
        os.mkdir(os.path.join(exp_path, 'figures'))
        os.mkdir(os.path.join(exp_path, 'towers'))
        os.mkdir(os.path.join(exp_path, 'models'))
        os.mkdir(os.path.join(exp_path, 'datasets'))
        os.mkdir(os.path.join(exp_path, 'val_datasets'))
        os.mkdir(os.path.join(exp_path, 'acquisition_data'))

        with open(os.path.join(exp_path, 'args.pkl'), 'wb') as handle:
            pickle.dump(args, handle)

        return ActiveExperimentLogger(exp_path, use_latents=args.use_latents)

    def save_dataset(self, dataset, tx):
        fname = 'active_%d.pkl' % tx
        with open(os.path.join(self.exp_path, 'datasets', fname), 'wb') as handle:
            pickle.dump(dataset, handle)

    def save_particles(self, particles, tx):
        fname = 'particles_%d.pkl' % tx
        with open(os.path.join(self.exp_path, 'models', fname), 'wb') as handle:
            pickle.dump(particles, handle)

    def load_particles(self, tx):
        fname = 'particles_%d.pkl' % tx
        with open(os.path.join(self.exp_path, 'models', fname), 'rb') as handle:
            particles = pickle.load(handle)
        return particles
            
    def save_val_dataset(self, val_dataset, tx):
        fname = 'val_active_%d.pkl' % tx
        with open(os.path.join(self.exp_path, 'val_datasets', fname), 'wb') as handle:
            pickle.dump(val_dataset, handle)

    def load_dataset(self, tx):
        fname = 'active_%d.pkl' % tx 
        path = os.path.join(self.exp_path, 'datasets', fname)
        try:
            with open(path, 'rb') as handle:
                dataset = pickle.load(handle)
            return dataset
        except:
            print('active_%d.pkl not found on path' % tx)
            return None
            
    def load_val_dataset(self, tx):
        fname = 'val_active_%d.pkl' % tx 
        path = os.path.join(self.exp_path, 'val_datasets', fname)
        try:
            with open(path, 'rb') as handle:
                val_dataset = pickle.load(handle)
            return val_dataset
        except:
            print('val_active_%d.pkl not found on path' % tx)
            return None

    def get_figure_path(self, fname):
        if not os.path.exists(os.path.join(self.exp_path, 'figures')): 
            os.mkdir(os.path.join(self.exp_path, 'figures'))
        return os.path.join(self.exp_path, 'figures', fname)

    def get_towers_path(self, fname):
        return os.path.join(self.exp_path, 'towers', fname)

    def get_ensemble(self, tx):
        """ Load an ensemble from the logging structure.
        :param tx: The active learning iteration of which ensemble to load.
        :return: learning.models.Ensemble object.
        """
        # Load metadata and initialize ensemble.
        path = os.path.join(self.exp_path, 'models', 'metadata.pkl')
        with open(path, 'rb') as handle:
            metadata = pickle.load(handle)
        ensemble = Ensemble(base_model=metadata['base_model'],
                            base_args=metadata['base_args'],
                            n_models=metadata['n_models'])
        if self.use_latents:
            ensemble = LatentEnsemble(ensemble,
                n_latents=metadata['n_latents'],
                d_latents=metadata['d_latents'])

        # Load ensemble weights.
        path = os.path.join(self.exp_path, 'models', 'ensemble_%d.pt' % tx)
        try:
            ensemble.load_state_dict(torch.load(path, map_location='cpu'))
            return ensemble
        except:
            print('ensemble_%d.pkl not found on path' % tx)
            return None



    def save_ensemble(self, ensemble, tx):
        """ Save an ensemble within the logging directory. The weights
        will be saved to <exp_name>/models/ensemble_<tx>.pt. Model metadata that
        is needed to initialize the Ensemble class while loading is 
        save to <exp_name>/models/metadata.pkl.

        :ensemble: A learning.model.Ensemble object.
        :tx: The active learning timestep these models represent.
        """
        if self.use_latents:
            self.save_latent_ensemble(ensemble, tx)
            return

        # Save ensemble metadata.
        metadata = {'base_model': ensemble.base_model,
                    'base_args': ensemble.base_args,
                    'n_models': ensemble.n_models}
        path = os.path.join(self.exp_path, 'models', 'metadata.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(metadata, handle)

        # Save ensemble weights.
        path = os.path.join(self.exp_path, 'models', 'ensemble_%d.pt' % tx)
        torch.save(ensemble.state_dict(), os.path.join(path))

    def save_latent_ensemble(self, latent_ensemble, tx):
        metadata = {'base_model': latent_ensemble.ensemble.base_model,
                    'base_args': latent_ensemble.ensemble.base_args,
                    'n_models': latent_ensemble.ensemble.n_models,
                    'n_latents': latent_ensemble.n_latents,
                    'd_latents': latent_ensemble.d_latents}
        path = os.path.join(self.exp_path, 'models', 'metadata.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(metadata, handle)

        # Save ensemble weights.
        path = os.path.join(self.exp_path, 'models', 'ensemble_%d.pt' % tx)
        torch.save(latent_ensemble.state_dict(), os.path.join(path))

    def get_towers_data(self, tx):
        # Get all tower files at the current acquisition step, in sorted order
        tower_files = []
        all_towers = os.listdir(os.path.join(self.exp_path, 'towers'))
        for tower_file in all_towers:
            match_str = r'labeled_tower_(.*)_(.*)_(.*)_{}.pkl'.format(tx)
            if re.match(match_str, tower_file):
                tower_files.append(tower_file)
        tower_files.sort()
        # Extract the tower data from each tower file
        tower_data = []
        for tower_file in tower_files:
            with open(self.get_towers_path(tower_file), 'rb') as handle:
                tower_tx_data = pickle.load(handle)
            tower_data.append(tower_tx_data)
        return tower_data


    def save_towers_data(self, block_tower, block_ids, label):
        fname = 'labeled_tower_{:%Y-%m-%d_%H-%M-%S}_'.format(datetime.datetime.now())\
                +str(self.tower_counter)+'_'+str(self.acquisition_step)+'.pkl'
        path = self.get_towers_path(fname)
        with open(path, 'wb') as handle:
            pickle.dump([block_tower, block_ids, label], handle)
        self.tower_counter += 1

    def save_acquisition_data(self, new_data, samples, tx):
        data = {
            'acquired_data': new_data,
            'samples': samples
        }
        path = os.path.join(self.exp_path, 'acquisition_data', 'acquired_%d.pkl' % tx)
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)
        self.acquisition_step = tx+1
        self.tower_counter = 0
        self.remove_unlabeled_acquisition_data()
        
    def remove_unlabeled_acquisition_data(self):
        os.remove(os.path.join(self.exp_path, 'acquired_processing.pkl'))
        
    def save_unlabeled_acquisition_data(self, data):
        path = os.path.join(self.exp_path, 'acquired_processing.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)
            
    def get_unlabeled_acquisition_data(self):
        path = os.path.join(self.exp_path, 'acquired_processing.pkl')
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        return data
        
    def get_block_placement_data(self):
        path = os.path.join(self.exp_path, 'block_placement_data.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as handle:
                block_placements = pickle.load(handle)
            return block_placements
        else:
            return {}
            
    def save_block_placement_data(self, block_placements):
        block_placement_data = self.get_block_placement_data()
        block_placement_data[self.acquisition_step] = block_placements
        with open(os.path.join(self.exp_path, 'block_placement_data.pkl'), 'wb') as handle:
            pickle.dump(block_placement_data, handle)
    
    def load_acquisition_data(self, tx):
        path = os.path.join(self.exp_path, 'acquisition_data', 'acquired_%d.pkl' % tx)
        try:
            with open(path, 'rb') as handle:
                data = pickle.load(handle)
            return data['acquired_data'], data['samples']
        except:
            print('acquired_%d.pkl not found on path' % tx)
            return None, None
            
    def save_evaluation_tower(self, tower, reward, max_reward, tx, planning_model, task, noise=None):
        if noise:
            tower_file = 'towers_%d_%f.pkl' % (tx, noise)
        else:
            tower_file = 'towers_%d.pkl' % tx
        tower_height = len(tower)
        tower_key = '%dblock' % tower_height
        tower_path = os.path.join(self.exp_path, 'evaluation_towers', task, planning_model)
        if not os.path.exists(tower_path): 
            os.makedirs(tower_path)
        if not os.path.isfile(os.path.join(tower_path, tower_file)):
            towers = {}
            print('Saving evaluation tower to %s' % os.path.join(tower_path, tower_file))
        else:
            with open(os.path.join(tower_path, tower_file), 'rb') as f:
                towers = pickle.load(f)
        if tower_key in towers:
            towers[tower_key].append((tower, reward, max_reward))
        else:
            towers[tower_key] = [(tower, reward, max_reward)]
        with open(os.path.join(tower_path, tower_file), 'wb') as f:
            pickle.dump(towers, f)
        print('APPENDING evaluation tower to %s' % os.path.join(tower_path, tower_file))   
        
    def get_evaluation_towers(self, task, planning_model, tx):
        tower_path = os.path.join(self.exp_path, 'evaluation_towers', task, planning_model)
        towers_data = {}
        for file in os.listdir(tower_path):
            if file != '.DS_Store':
                with open(os.path.join(tower_path, file), 'rb') as f:
                    data = pickle.load(f)
                towers_data[os.path.join(tower_path, file)] = data
        return towers_data
        