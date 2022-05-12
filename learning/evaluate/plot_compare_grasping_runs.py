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
    output_path = os.path.join('learning/evaluate/grasping_comparisons', args.output_folder_name)
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
            logger = ActiveExperimentLogger(exp_path, use_latents=True)
            loggers[name].append(logger)
    return loggers

def plot_val_loss(loggers, output_path):
    plt.clf()
    fig, axes = plt.subplots(1, sharex=False, figsize=(20,10))
    val_fname = 'val_accuracies.pkl'
    # val_fname = 'val_precisions.pkl'
    # val_fname = 'val_f1s.pkl'
    for name, group_loggers in loggers.items():
        # First create one large results dictionary with pooled results from each logger.
        all_accs = []
        for logger in group_loggers:
            init, n_acquire = logger.get_acquisition_params()
            val_path = logger.get_figure_path(val_fname)
            if not os.path.exists(val_path):
                print('[WARNING] %s not evaluated.' % logger.exp_path)
                continue
            with open(val_path, 'rb') as handle:
                vals = pickle.load(handle)

            if len(all_accs) == 0:
                all_accs = [[acc] for acc in vals]
            else:
                for tx in range(len(all_accs)):
                    if tx >= len(vals):
                        all_accs[tx].append(vals[-1])
                    else:
                        all_accs[tx].append(vals[tx])
        if len(all_accs) == 0:
            continue
        # Then plot the regrets and save it in the results folder.
        upper75 = [] 
        median = [] 
        lower25 = [] 
        for tx in range(len(all_accs)):
            median.append(np.median(all_accs[tx]))
            lower25.append(np.quantile(all_accs[tx], 0.25))
            upper75.append(np.quantile(all_accs[tx], 0.75))

        xs = np.arange(init, init+len(median), n_acquire)
        # if 'ensemble' in name:
        #     print(name)
        #     xs = xs *3
        axes.plot(xs, median, label=name)
        axes.fill_between(xs, lower25, upper75, alpha=0.2)
        axes.set_ylim(0.25, 1.1)
        axes.set_ylabel('Val Accuracy')
        axes.set_xlabel('Number of adaptation towers')
        axes.legend()
    # plt_fname = 'validation_accuracy.png'
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-groups', nargs='+', type=str, default=[])
    parser.add_argument('--eval-type', type=str, choices=['val'])
    parser.add_argument('--output-folder-name', type=str)
    parser.add_argument('--comp-name', choices=['test-accuracy'], required=True)
    args = parser.parse_args()
    print(args)

    output_path = create_output_dir(args)

    loggers = get_loggers_from_run_groups(args.run_groups)
    
    if args.eval_type == 'val':
        plot_val_loss(loggers, output_path)
    else:
        assert(len(args.problem) > 1)
        plot_task_regret(loggers, args.problem, output_path, args.comp_name)



