import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

from learning.active.utils import ActiveExperimentLogger
from learning.domains.towers.active_utils import get_sequential_predictions, get_predictions

tower_heights = [2, 3, 4, 5, 6, 7]
min_towers_acq = 40         # number of towers in initial dataset
towers_per_acq = 10         # number of towers acquired between each trained model
                
def calc_model_accuracy(logger, dataset, args, exp_path, save_local_fig=True):
    if args.max_acquisitions is not None: 
        eval_range = range(0, args.max_acquisitions, args.plot_step)
    elif args.single_acquisition_step is not None: 
        eval_range = [args.single_acquisition_step]

    accuracies = {key: [] for key in dataset.keys()}
    false_positives = {key: [] for key in dataset.keys()}
    false_negatives = {key: [] for key in dataset.keys()}
    for tx in eval_range:
        print('Acquisition step '+str(tx))
        ensemble = logger.get_ensemble(tx)
        if torch.cuda.is_available():
            ensemble = ensemble.cuda()
        if logger.args.sampler == 'sequential' or logger.args.strategy == 'subtower' or logger.args.strategy == 'subtower-greedy':
            preds = get_sequential_predictions(dataset, ensemble)
            preds = preds.round()
        else:
            preds = get_predictions(dataset, ensemble)
            preds = preds.mean(axis=1).round()        
        
        samples_per_height = dataset[list(dataset.keys())[0]]['towers'].shape[0]
        
        for ti, tower_height in enumerate(tower_heights):
            key = str(tower_height)+'block'
            if key in dataset:
                n_correct = 0
                n_tn = 0
                n_fp = 0
                n_fn = 0
                n_tp = 0
                offset = ti*samples_per_height
                for li, label in enumerate(dataset[key]['labels']):
                    if preds[offset+li] == label[0]: n_correct += 1
                    if preds[offset+li] == 0 and label[0] == 1: n_fp += 1
                    if preds[offset+li] == 1 and label[0] == 0: n_fn += 1
                    if preds[offset+li] == 0 and label[0] == 0: n_tn += 1
                    if preds[offset+li] == 1 and label[0] == 1: n_tp += 1                
                accuracies[key].append(n_correct/samples_per_height)
                fpr = 0.0 if (n_fp+n_tn) == 0.0 else n_fp/(n_fp+n_tn)
                fnr = 0.0 if (n_fn+n_tp) == 0.0 else n_fn/(n_fn+n_tp)
                false_positives[key].append(fpr)
                false_negatives[key].append(fnr)
                
    # plot and save to this exp_path
    #for result, title in zip([accuracies, false_positives, false_negatives], ['Constructability Accuracy', 'False Positive Rate', 'False Negative Rate']):
    if save_local_fig:
        acquisition_plot_steps = len(range(0, args.max_acquisitions, args.plot_step))
        xs = np.arange(min_towers_acq, \
                        min_towers_acq+towers_per_acq*args.plot_step*acquisition_plot_steps, \
                        towers_per_acq*args.plot_step) # number of training towers
        fig, axes = plt.subplots(4, figsize=(5,12))
        for ki, key in enumerate(accuracies):
            axes[ki].plot(xs, accuracies[key], label='accuracy')
            #axes[ki].plot(xs, false_positives[key], label='false positive rate')
            #axes[ki].plot(xs, false_negatives[key], label='false negative rate')
            #axes[ki].set_ylim(.0, 1.)
            axes[ki].set_ylim(.5, 1.)
            axes[ki].set_ylabel('Rate')
            axes[ki].set_xlabel('Training Towers')
            axes[ki].set_title('%s Tower Constructability Accuracy' % key)
            axes[ki].legend()
        plt.tight_layout()
        plt_fname = 'constructability_accuracy.png'
        plt.savefig(logger.get_figure_path(plt_fname))
        plt.close()
    return accuracies
    
def plot_all_model_accuracies(all_model_accuracies):
    fig, axes = plt.subplots(4, figsize=(5,12))
    
    acquisition_plot_steps = len(range(0, args.max_acquisitions, args.plot_step))
    xs = np.arange(min_towers_acq, \
                    min_towers_acq+towers_per_acq*args.plot_step*acquisition_plot_steps, \
                    towers_per_acq*args.plot_step) # number of training towers

    for pi, th in enumerate(tower_heights):
        for model_accuracies in all_model_accuracies:
            key = str(th)+'block'
            if key in model_accuracies:
                pi_accuracies = model_accuracies[key]
                axes[pi].plot(xs, pi_accuracies)
                axes[pi].set_ylim(.5, 1.)
                axes[pi].set_ylabel('Constructability Accuracy')
                axes[pi].set_xlabel('Training Towers')
                axes[pi].set_title(str(th)+' Block Tower Constructability Accuracy')
        
    plt.tight_layout()
    plt.savefig(args.output_fname)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-paths', 
                        nargs='+', 
                        required=True)
    parser.add_argument('--max-acquisitions',
                        type=int, 
                        help='evaluate from 0 to this acquisition step (use either this or --acquisition-step)')
    parser.add_argument('--plot-step',
                        type=int,
                        default=10)
    parser.add_argument('--single-acquisition-step',
                        type=int,
                        help='evaluate only this acquisition step(use either this or --max-acquisitions)')
    parser.add_argument('--test-set-fname',
                        type=str,
                        required=True,
                        help='evaluate only this acquisition step(use either this or --max-acquisitions)')                        
    parser.add_argument('--output-fname',
                        type=str,
                        required=True,
                        help='evaluate only this acquisition step(use either this or --max-acquisitions)')                        
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    
    # TODO: cannot do false positive and negative rates wit multiple exp paths
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    with open(args.test_set_fname, 'rb') as f:
        dataset = pickle.load(f)
        
    all_accuracies = []
    for exp_path in args.exp_paths:
        logger = ActiveExperimentLogger(exp_path)
        model_accuracies = calc_model_accuracy(logger, dataset, args, exp_path)
        all_accuracies.append(model_accuracies)
        
    if args.single_acquisition_step is None:
        plot_all_model_accuracies(all_accuracies)
    else:
        print('Accuracy per model: ')
        for i, acc in enumerate(all_accuracies):
            print('    Model '+str(i)+':', acc)