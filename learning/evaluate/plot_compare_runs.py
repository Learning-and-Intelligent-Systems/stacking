import argparse
import numpy as np
import os
import pickle
import sys

from learning.active.utils import ActiveExperimentLogger

import matplotlib
from matplotlib import pyplot as plt
matplotlib.rc('font', family='normal', size=28)


NAMES = {
    'ensemble-comp': {
        '50_noens_pf_fit': 'No ensemble',
        '50_norot_pf_active_long': '7-model Ensemble'
    },
    'adapt-comp': {
        '50_ensemble_active': 'Baseline Deep Ensemble Adaptation',
        '50_norot_pf_random': 'Random PF Adaptation',
        '50_norot_pf_active_long': 'Active PF Adaptation'
    }
}

COLORS = {
    'ensemble-comp': {
        '50_noens_pf_fit': 'r',
        '50_norot_pf_active_long': 'g'
    },
    'adapt-comp': {
        '50_ensemble_active': 'r',
        '50_norot_pf_random': 'b',
        '50_norot_pf_active_long': 'g'
    }
}

def create_output_dir(args):
    output_path = os.path.join('learning/evaluate/comparisons', args.output_folder_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        with open(os.path.join(output_path, 'args.pkl'), 'rb') as handle:
            old_args = pickle.load(handle)
        if set(args.run_groups) != set(old_args.run_groups):
            print('[ERROR] --run-groups must match those aleady used in this folder')
            sys.exit(0)

    with open(os.path.join(output_path, 'args.pkl'), 'wb') as handle:
        pickle.dump(args, handle)
    return output_path


def get_loggers_from_run_groups(run_groups):
    loggers = {}
    for fname in run_groups:
        name = fname.split('/')[-1].split('.')[0]
        loggers[name] = []
        with open(fname, 'r') as handle:
            exp_paths = [l.strip() for l in handle.readlines()]
        for exp_path in exp_paths:
            logger = ActiveExperimentLogger(os.path.join('learning/experiments/logs', exp_path), use_latents=True)
            loggers[name].append(logger)
    return loggers


def plot_task_regret(loggers, problem, output_path, comp_name):
    tower_keys = ['2block']
    fig, axes = plt.subplots(len(tower_keys), sharex=True, figsize=(20,10))
    axes = [axes]
    regret_fname = '%s_-1.00_regrets.pkl' % problem
    for name, group_loggers in loggers.items():
        # First create one large results dictionary with pooled results from each logger.
        all_regrets = {}
        for logger in group_loggers:
            init, n_acquire = logger.get_acquisition_params()
            regret_path = logger.get_figure_path(regret_fname)
            if not os.path.exists(regret_path):
                print('[WARNING] %s not evaluated.' % logger.exp_path)
                continue
            with open(regret_path, 'rb') as handle:
                regrets = pickle.load(handle)
            for k in regrets:
                if k not in all_regrets:
                    all_regrets[k] = regrets[k]
                else:
                    for tx in range(len(regrets[k])):
                        all_regrets[k][tx] += regrets[k][tx]
        
        # Then plot the regrets and save it in the results folder.
        if comp_name == 'ensemble-comp':
            max_tx = 10
        else:
            max_tx = 20
        upper75 = {k: [] for k in tower_keys}
        median = {k: [] for k in tower_keys}
        lower25 = {k: [] for k in tower_keys}
        for k in tower_keys:
            rs = all_regrets[k]
            for tx in range(max_tx):
                median[k].append(np.median(rs[tx]))
                lower25[k].append(np.quantile(rs[tx], 0.25))
                upper75[k].append(np.quantile(rs[tx], 0.75))

        for kx, k in enumerate(tower_keys):
            xs = np.arange(init, init+2*len(median[k]), 2*n_acquire)
            if 'ensemble' in name:
                print(name)
                xs = xs * 3

            axes[kx].plot(xs, median[k], label=NAMES[comp_name][name], c=COLORS[comp_name][name])
            axes[kx].fill_between(xs, lower25[k], upper75[k], alpha=0.1, color=COLORS[comp_name][name])
            axes[kx].set_xlim(0, max_tx*2-2)
            axes[kx].set_ylim(0.0, 1.1)
            axes[kx].set_ylabel('Regret (Normalized)')
            axes[kx].set_xlabel('Number of adaptation towers')
            axes[kx].legend()
    plt.tight_layout()
    plt_fname = comp_name + problem + '.png'
    plt.savefig(os.path.join(output_path, plt_fname))

def plot_val_loss(loggers, output_path):
    tower_keys = ['2block', '3block', '4block', '5block']
    fig, axes = plt.subplots(len(tower_keys), sharex=False, figsize=(10,20))
    val_fname = 'val_accuracies.pkl'
    for name, group_loggers in loggers.items():
        # First create one large results dictionary with pooled results from each logger.
        all_accs = {}
        for logger in group_loggers:
            init, n_acquire = logger.get_acquisition_params()
            val_path = logger.get_figure_path(val_fname)
            if not os.path.exists(val_path):
                print('[WARNING] %s not evaluated.' % logger.exp_path)
                continue
            with open(val_path, 'rb') as handle:
                vals = pickle.load(handle)
            for k in vals:
                if k not in all_accs:
                    all_accs[k] = [[acc] for acc in vals[k]]
                else:
                    for tx in range(len(all_accs[k])):
                        all_accs[k][tx].append(vals[k][tx])
        
        # Then plot the regrets and save it in the results folder.
        upper75 = {k: [] for k in tower_keys}
        median = {k: [] for k in tower_keys}
        lower25 = {k: [] for k in tower_keys}
        for k in tower_keys:
            rs = all_accs[k]
            for tx in range(len(rs)):
                median[k].append(np.median(rs[tx]))
                lower25[k].append(np.quantile(rs[tx], 0.25))
                upper75[k].append(np.quantile(rs[tx], 0.75))

        for kx, k in enumerate(tower_keys):
            xs = np.arange(init, init+len(median[k]), n_acquire)
            if 'ensemble' in name:
                print(name)
                xs = xs *3
            axes[kx].plot(xs, median[k], label=name)
            axes[kx].fill_between(xs, lower25[k], upper75[k], alpha=0.2)
            axes[kx].set_ylim(0.5, 1.1)
            axes[kx].set_ylabel('Val Accuracy')
            axes[kx].set_xlabel('Number of adaptation towers')
            axes[kx].legend()
    plt_fname = 'validation_accuracy.png'
    plt.savefig(os.path.join(output_path, plt_fname))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-groups', nargs='+', type=str, default=[])
    parser.add_argument('--eval-type', type=str, choices=['task', 'val'])
    parser.add_argument('--problem', default='', type=str, choices=['any-overhang', 'min-contact', 'cumulative-overhang', 'tallest'])
    parser.add_argument('--output-folder-name', type=str)
    parser.add_argument('--comp-name', choices=['ensemble-comp', 'adapt-comp'], required=True)
    args = parser.parse_args()
    print(args)

    output_path = create_output_dir(args)

    loggers = get_loggers_from_run_groups(args.run_groups)
    
    if args.eval_type == 'val':
        plot_val_loss(loggers, output_path)
    else:
        assert(len(args.problem) > 1)
        plot_task_regret(loggers, args.problem, output_path, args.comp_name)



