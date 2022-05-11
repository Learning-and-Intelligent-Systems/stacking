import argparse
import json
import numpy as np
import os
import pickle
import sys

from learning.active.utils import ActiveExperimentLogger
from learning.evaluate.evaluate_grasping import get_pf_validation_accuracy
from learning.evaluate.plot_compare_grasping_runs import plot_val_loss
from learning.experiments.train_grasping_single import run as training_phase
from learning.experiments.active_fit_grasping_pf import run_particle_filter_fitting as fitting_phase


DATA_ROOT = 'learning/data/grasping'
EXPERIMENT_ROOT = 'learning/experiments/metadata'

def get_dataset_path_or_fail(args):
    if len(args.dataset_name) == 0:
        print(f'[ERROR] You must specify a dataset.')
        sys.exit()
    dataset_dir = os.path.join(DATA_ROOT, args.dataset_name)
    if not os.path.exists(dataset_dir):
        print(f'[ERROR] Dataset does not exist: {args.dataset_name}')
        sys.exit()
    return dataset_dir

def create_experiment(args):
    exp_dir = os.path.join(EXPERIMENT_ROOT, args.exp_name)
    if os.path.exists(exp_dir):
        print(f'[ERROR] Folder already exists: {exp_dir}')
        sys.exit()
    
    dataset_dir = get_dataset_path_or_fail(args)
    os.makedirs(exp_dir)

    args_path = os.path.join(exp_dir, 'args.pkl')
    with open(args_path, 'wb') as handle:
        pickle.dump(args, handle)
    
    logs_path = os.path.join(exp_dir, 'logs_lookup.json')
    logs = {
        'training_phase': '',
        'fitting_phase': {'random': {}, 'bald': {}}
    }
    with open(logs_path, 'w') as handle:
        json.dump(logs, handle)

    log_groups_path = os.path.join(exp_dir, 'log_groups')
    os.makedirs(log_groups_path)


def get_training_phase_dataset_args(dataset_fname):
    data_path = os.path.join(DATA_ROOT, dataset_fname)

    train_path = os.path.join(data_path, 'grasps', 'training_phase', 'train_grasps.pkl')
    val_path = os.path.join(data_path, 'grasps', 'training_phase', 'val_grasps.pkl')

    with open(train_path, 'rb') as handle:
        train_grasps = pickle.load(handle)
    n_objects = len(train_grasps['object_data']['object_names'])

    return train_path, val_path, n_objects

def get_fitting_phase_dataset_args(dataset_fname):
    data_path = os.path.join(DATA_ROOT, dataset_fname)

    train_geo_fname = os.path.join(data_path, 'objects', 'train_geo_test_props.pkl')
    test_geo_fname = os.path.join(data_path, 'objects', 'test_geo_test_props.pkl')

    with open(train_geo_fname, 'rb') as handle:
        train_geo_objects = pickle.load(handle)
    n_train_geo = len(train_geo_objects['object_data']['object_names'])

    with open(test_geo_fname, 'rb') as handle:
        test_geo_objects = pickle.load(handle)
    n_test_geo = len(test_geo_objects['object_data']['object_names'])

    return train_geo_fname, test_geo_fname, n_train_geo, n_test_geo

def run_fitting_phase(args):
    exp_path = os.path.join(EXPERIMENT_ROOT, args.exp_name)
    if not os.path.exists(exp_path):
        print(f'[ERROR] Experiment does not exist: {args.exp_name}')
        sys.exit()

    logs_path = os.path.join(exp_path, 'logs_lookup.json')
    with open(logs_path, 'r') as handle:
        logs_lookup = json.load(handle)

    pretrained_model_path = logs_lookup['training_phase']
    if len(pretrained_model_path) == 0:
        print(f'[ERROR] Training phase has not yet been executed.')
        sys.exit()

    # Get train_geo_test_props.pkl and test_geo_test_props.pkl
    args_path = os.path.join(exp_path, 'args.pkl')
    with open(args_path, 'rb') as handle:
        exp_args = pickle.load(handle)

    train_geo_fname, test_geo_fname, n_train_geo, n_test_geo = get_fitting_phase_dataset_args(exp_args.dataset_name)

    # Run fitting phase for all objects that have not yet been evaluated (each has a standard name in the experiment logs).
    
    for geo_type, objects_fname, n_objects in zip(['train_geo', 'test_geo'], [train_geo_fname, test_geo_fname], [n_train_geo, n_test_geo]): 
        
        for ox in range(n_objects):
            if ox > 100:
                print(ox)
                break
            fitting_exp_name = f'grasp_{exp_args.exp_name}_fit_{args.strategy}_{geo_type}_object{ox}'

            # Check if we have already fit this object.
            if fitting_exp_name in logs_lookup['fitting_phase'][args.strategy]:
                print(f'Skipping {fitting_exp_name}...')
                continue

            fitting_args = argparse.Namespace()
            fitting_args.exp_name = fitting_exp_name
            fitting_args.max_acquisitions = 10
            fitting_args.objects_fname = objects_fname
            fitting_args.n_samples = 50
            fitting_args.pretrained_ensemble_exp_path = pretrained_model_path
            fitting_args.ensemble_tx = 0
            fitting_args.eval_object_ix = ox
            fitting_args.strategy = args.strategy
            fitting_args.n_particles = 200

            print(f'Running fitting phase: {fitting_exp_name}')
            fit_log_path = fitting_phase(fitting_args)

            # Save fitting path in metadata.
            if len(logs_lookup['fitting_phase']) == 0:
                logs_lookup['fitting_phase'] = {'random': {}, 'bald': {}}
            logs_lookup['fitting_phase'][args.strategy][fitting_exp_name] = fit_log_path
            with open(logs_path, 'w') as handle:
                json.dump(logs_lookup, handle)

            # Run accuracy evaluations for this object.
            print(f'Evaluating fitting phase: {fitting_exp_name}')
            fit_logger = ActiveExperimentLogger(fit_log_path, use_latents=True) 
            val_dataset_fname = f'fit_grasps_{geo_type}_object{ox}.pkl'
            val_dataset_path = os.path.join(DATA_ROOT, exp_args.dataset_name, 'grasps', 'fitting_phase', val_dataset_fname)

            get_pf_validation_accuracy(fit_logger, val_dataset_path)

            


def run_training_phase(args):
    exp_path = os.path.join(EXPERIMENT_ROOT, args.exp_name)
    if not os.path.exists(exp_path):
        print(f'[ERROR] Experiment does not exist: {args.exp_name}')
        sys.exit()

    args_path = os.path.join(exp_path, 'args.pkl')
    with open(args_path, 'rb') as handle:
        exp_args = pickle.load(handle)

    logs_path = os.path.join(exp_path, 'logs_lookup.json')
    with open(logs_path, 'r') as handle:
        logs_lookup = json.load(handle)
    if len(logs_lookup['training_phase']) > 0:
        print('[ERROR] Model already trained.')
        sys.exit()

    train_data_fname, val_data_fname, n_objs = get_training_phase_dataset_args(exp_args.dataset_name)

    training_args = argparse.Namespace()
    training_args.exp_name = f'grasp_{exp_args.exp_name}_train'
    training_args.train_dataset_fname = train_data_fname
    training_args.val_dataset_fname = val_data_fname
    training_args.n_objects = n_objs
    training_args.n_epochs = 20
    training_args.model = 'pn'
    training_args.n_hidden = 64
    training_args.batch_size = 16
    training_args.property_repr = 'latent'
    training_args.n_models = 5

    train_log_path = training_phase(training_args)
    
    # Save training path in metadata.
    logs_lookup['training_phase'] = train_log_path
    with open(logs_path, 'w') as handle:
        json.dump(logs_lookup, handle)
    

def run_testing_phase(args):
    
    # Create log_group files.
    exp_path = os.path.join(EXPERIMENT_ROOT, args.exp_name)
    if not os.path.exists(exp_path):
        print(f'[ERROR] Experiment does not exist: {args.exp_name}')
        sys.exit()

    logs_path = os.path.join(exp_path, 'logs_lookup.json')
    with open(logs_path, 'r') as handle:
        logs_lookup = json.load(handle)

    # Get train_geo_test_props.pkl and test_geo_test_props.pkl
    args_path = os.path.join(exp_path, 'args.pkl')
    with open(args_path, 'rb') as handle:
        exp_args = pickle.load(handle)

    # Get object data.
    train_objects_fname = os.path.join(DATA_ROOT, exp_args.dataset_name, 'objects', 'train_geo_test_props.pkl')
    with open(train_objects_fname, 'rb') as handle:
        train_objects = pickle.load(handle)
    test_objects_fname = os.path.join(DATA_ROOT, exp_args.dataset_name, 'objects', 'test_geo_test_props.pkl')
    with open(test_objects_fname, 'rb') as handle:
        test_objects = pickle.load(handle)

    # Nested dictionary: [train_geo/test_geo][random/bald][all/object_name]
    logs_lookup_by_object = {
        'train_geo': {
            'random': {
                'all': [],
            },
            'bald': {
                'all': [],
            }
        },
        'test_geo': {
            'random': {
                'all': [],
            },
            'bald': {
                'all': [],
            }
        }
    }

    n_found = 0
    p_stable_low, p_stable_high = 0.0, 1.0
    for ox, object_name in enumerate(train_objects['object_data']['object_names']):
        # TO REMOVE. (2 lines)
        val_dataset_fname = f'fit_grasps_train_geo_object{ox}.pkl'
        val_dataset_path = os.path.join(DATA_ROOT, exp_args.dataset_name, 'grasps', 'fitting_phase', val_dataset_fname)

        with open(val_dataset_path, 'rb') as handle:
            data = pickle.load(handle)
            p_stable = np.mean(data['grasp_data']['labels'])
            if p_stable < p_stable_low or p_stable > p_stable_high:
                continue
            n_found += 1
        print(f'{object_name} in range ({p_stable_low}, {p_stable_high}) ({p_stable})')

        if object_name not in logs_lookup_by_object['train_geo']['random']:
            logs_lookup_by_object['train_geo']['random'][object_name] = []
        if object_name not in logs_lookup_by_object['train_geo']['bald']:
            logs_lookup_by_object['train_geo']['bald'][object_name] = []

        random_log_key = f'grasp_{exp_args.exp_name}_fit_random_train_geo_object{ox}'
        if random_log_key in logs_lookup['fitting_phase']['random']:
            random_log_fname = logs_lookup['fitting_phase']['random'][random_log_key]

            logs_lookup_by_object['train_geo']['random']['all'].append(random_log_fname)
            logs_lookup_by_object['train_geo']['random'][object_name].append(random_log_fname)
            
            # TO REMVOE (2 lines)
            # fit_logger = ActiveExperimentLogger(random_log_fname, use_latents=True) 
            # get_pf_validation_accuracy(fit_logger, val_dataset_path)


        bald_log_key = f'grasp_{args.exp_name}_fit_bald_train_geo_object{ox}'
        if bald_log_key in logs_lookup['fitting_phase']['bald']:
            bald_log_fname = logs_lookup['fitting_phase']['bald'][bald_log_key]

            logs_lookup_by_object['train_geo']['bald']['all'].append(bald_log_fname)
            logs_lookup_by_object['train_geo']['bald'][object_name].append(bald_log_fname)

            # TO REMOVE (2 lines)
            # fit_logger = ActiveExperimentLogger(bald_log_fname, use_latents=True) 
            # get_pf_validation_accuracy(fit_logger, val_dataset_path)

        if ox > 100:
            break
    print(f'{n_found} train geo objects included.')
    n_found = 0
    for ox, object_name in enumerate(test_objects['object_data']['object_names']):
         # TO REMOVE. (2 lines)
        val_dataset_fname = f'fit_grasps_test_geo_object{ox}.pkl'
        val_dataset_path = os.path.join(DATA_ROOT, exp_args.dataset_name, 'grasps', 'fitting_phase', val_dataset_fname)

        with open(val_dataset_path, 'rb') as handle:
            data = pickle.load(handle)
            p_stable = np.mean(data['grasp_data']['labels'])
            if p_stable < p_stable_low or p_stable > p_stable_high:
                continue
            n_found += 1
        
        if object_name not in logs_lookup_by_object['test_geo']['random']:
            logs_lookup_by_object['test_geo']['random'][object_name] = []
        if object_name not in logs_lookup_by_object['test_geo']['bald']:
            logs_lookup_by_object['test_geo']['bald'][object_name] = []

        random_log_key = f'grasp_{exp_args.exp_name}_fit_random_test_geo_object{ox}'
        if random_log_key in logs_lookup['fitting_phase']['random']:
            random_log_fname = logs_lookup['fitting_phase']['random'][random_log_key]

            logs_lookup_by_object['test_geo']['random']['all'].append(random_log_fname)
            logs_lookup_by_object['test_geo']['random'][object_name].append(random_log_fname)

            # TO REMVOE (2 lines)
            # fit_logger = ActiveExperimentLogger(random_log_fname, use_latents=True) 
            # get_pf_validation_accuracy(fit_logger, val_dataset_path)


        bald_log_key = f'grasp_{args.exp_name}_fit_bald_test_geo_object{ox}'
        if bald_log_key in logs_lookup['fitting_phase']['bald']:
            bald_log_fname = logs_lookup['fitting_phase']['bald'][bald_log_key]
            
            logs_lookup_by_object['test_geo']['bald']['all'].append(bald_log_fname)
            logs_lookup_by_object['test_geo']['bald'][object_name].append(bald_log_fname)

            # TO REMVOE (2 lines)
            # fit_logger = ActiveExperimentLogger(bald_log_fname, use_latents=True) 
            # get_pf_validation_accuracy(fit_logger, val_dataset_path)
    print(f'{n_found} test geo objects included.')

    for  obj_name, loggers in logs_lookup_by_object['train_geo']['random'].items():
        all_train_loggers = {
            f'{obj_name}_traingeo_random': [ActiveExperimentLogger(exp_path=name, use_latents=True) for name in loggers],
            f'{obj_name}_traingeo_bald': [ActiveExperimentLogger(exp_path=name, use_latents=True) for name in logs_lookup_by_object['train_geo']['bald'][obj_name]]
        }
        fig_path = os.path.join(exp_path, 'figures', f'{obj_name}_traingeo.png')
        plot_val_loss(all_train_loggers, fig_path)
    
    for  obj_name, loggers in logs_lookup_by_object['test_geo']['random'].items():
        all_test_loggers = {
            f'{obj_name}_testgeo_random': [ActiveExperimentLogger(exp_path=name, use_latents=True) for name in loggers],
            f'{obj_name}_testgeo_bald': [ActiveExperimentLogger(exp_path=name, use_latents=True) for name in logs_lookup_by_object['test_geo']['bald'][obj_name]]
        }
        fig_path = os.path.join(exp_path, 'figures', f'{obj_name}_testgeo.png')
        plot_val_loss(all_test_loggers, fig_path)


parser = argparse.ArgumentParser()
parser.add_argument('--phase', required=True, choices=['create', 'training', 'fitting', 'testing'])
parser.add_argument('--dataset-name', type=str, default='')
parser.add_argument('--exp-name', required=True, type=str)
parser.add_argument('--strategy', type=str, choices=['bald', 'random'], default='random')
args = parser.parse_args()


if __name__ == '__main__':
    
    if args.phase == 'create':
        create_experiment(args)
    elif args.phase == 'training':
        run_training_phase(args)
    elif args.phase == 'fitting':
        run_fitting_phase(args)
    elif args.phase == 'testing':
        run_testing_phase(args)