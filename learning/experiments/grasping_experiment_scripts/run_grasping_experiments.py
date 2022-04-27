import argparse
import json
import os
import pickle
import sys

from learning.experiments.train_grasping_single import run as training_phase


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
        'fitting_phase': []
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
    

parser = argparse.ArgumentParser()
parser.add_argument('--phase', required=True, choices=['create', 'training', 'fitting', 'testing'])
parser.add_argument('--dataset-name', type=str, default='')
parser.add_argument('--exp-name', required=True, type=str)
args = parser.parse_args()


if __name__ == '__main__':
    
    if args.phase == 'create':
        create_experiment(args)
    elif args.phase == 'training':
        run_training_phase(args)
    elif args.phase == 'fitting':
        pass
    elif args.phase == 'testing':
        pass