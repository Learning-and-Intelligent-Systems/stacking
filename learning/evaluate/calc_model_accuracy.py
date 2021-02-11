import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

from learning.active.utils import ActiveExperimentLogger
from learning.domains.towers.active_utils import get_predictions

tower_heights = [2, 3, 4, 5]
min_towers_acq = 40         # number of towers in initial dataset
towers_per_acq = 10         # number of towers acquired between each trained model
                
def calc_model_accuracies(logger, dataset, args):
    if args.max_acquisitions is not None: 
        eval_range = range(0, args.max_acquisitions, args.plot_step)
    elif args.acquisition_step is not None: 
        eval_range = [args.acquisition_step]

    accuracies = {key: [] for key in dataset.keys()}
    for tx in eval_range:
        print('Acquisition step '+str(tx))
        ensemble = logger.get_ensemble(tx)
        preds = get_predictions(dataset, ensemble)
        preds = preds.mean(axis=1).round()
        
        samples_per_height = int(preds.shape[0]/4) # all preds are in a 1D array
    
        for ti, tower_height in enumerate(tower_heights):
            key = str(tower_height)+'block'
            n_correct = 0
            offset = ti*samples_per_height
            for li, label in enumerate(dataset[key]['labels']):
                if preds[offset+li] != label[0]: n_correct += 1
            accuracies[key].append(n_correct/samples_per_height)
        
    return accuracies
    
def plot_all_model_accuracies(all_model_accuracies):
    fig, axes = plt.subplots(4, figsize=(5,12))
    
    acquisition_plot_steps = len(range(0, args.max_acquisitions, args.plot_step))
    xs = np.arange(min_towers_acq, \
                    min_towers_acq+towers_per_acq*args.plot_step*acquisition_plot_steps, \
                    towers_per_acq*args.plot_step) # number of training towers

    for pi, th in enumerate(tower_heights):
        for model_accuracies in all_model_accuracies:
            pi_accuracies = model_accuracies[str(th)+'block']
            axes[pi].plot(xs, pi_accuracies)
            axes[pi].set_ylim(0, .5)
            axes[pi].set_ylabel('Constructability Accuracy')
            axes[pi].set_xlabel('Training Towers')
            axes[pi].set_title(str(th)+' Block Tower Constructability Accuracy')
        
    plt_fname = 'constructability_accuracy.png'
    plt.tight_layout()
    plt.savefig('learning/evaluate/constructability_accuracy.png')
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
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    test_set_fname = 'learning/evaluate/test_constructability_dataset.pkl'
    with open(test_set_fname, 'rb') as f:
        dataset = pickle.load(f)
        
    all_accuracies = []
    for exp_path in args.exp_paths:
        logger = ActiveExperimentLogger(exp_path)
        model_accuracies = calc_model_accuracies(logger, dataset, args)
        all_accuracies.append(model_accuracies)
    plot_all_model_accuracies(all_accuracies)