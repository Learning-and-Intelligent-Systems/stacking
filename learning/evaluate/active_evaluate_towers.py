import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from learning.active.acquire import bald
from learning.active.utils import ActiveExperimentLogger
from learning.domains.towers.active_utils import get_predictions, sample_unlabeled_data


def get_validation_accuracy(logger, fname):
    tower_keys = ['2block', '3block', '4block', '5block']
    accs = {k: [] for k in tower_keys}

    with open(fname, 'rb') as handle:
        val_towers = pickle.load(handle)

    for tx in range(logger.args.max_acquisitions):
        if tx % 10 == 0:
            print('Eval timestep, ', tx)
        ensemble = logger.get_ensemble(tx)
        preds = get_predictions(val_towers, ensemble).mean(1).numpy()

        start = 0
        for k in tower_keys:
            end = start + val_towers[k]['towers'].shape[0]
            acc = ((preds[start:end]>0.5) == val_towers[k]['labels']).mean()
            accs[k].append(acc)
            start = end
    
    return accs


def get_dataset_statistics(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    for tx in range(logger.args.max_acquisitions):
        acquired_data, _ = logger.load_acquisition_data(tx)
        
        nums = [acquired_data[k]['towers'].shape[0] for k in tower_keys]
        pos = [np.sum(acquired_data[k]['labels']) for k in tower_keys]
        print(nums, pos)

    print('Totals')
    for tx in range(logger.args.max_acquisitions):
        dataset = logger.load_dataset(tx)
        print([dataset.tower_tensors[k].shape[0] for k in tower_keys])


def analyze_single_dataset(logger):
    tx = 200
    tower_keys = ['2block', '3block', '4block', '5block']
    dataset = logger.load_dataset(tx)

    for k in tower_keys:
        pos = np.sum(dataset.tower_labels[k].numpy())
        neg = dataset.tower_labels[k].shape[0] - pos

        print(k, '+', pos, '-', neg)



def analyze_bald_scores(logger):
    tower_keys = ['2block', '3block', '4block', '5block']

    tx = 226
    acquired, unlabeled = logger.load_acquisition_data(tx)
    ensemble = logger.get_ensemble(tx)

    preds = get_predictions(unlabeled, ensemble)
    bald_scores = bald(preds).numpy()
    acquire_indices = np.argsort(bald_scores)[::-1][:10]
    print(acquire_indices)
    start = 0
    for k in tower_keys:
        end = start + unlabeled[k]['towers'].shape[0]
        acquire_indices = np.argsort(bald_scores[start:end])[::-1][:25]
        print(bald_scores[start:end][acquire_indices])
        print(acquire_indices + start)
        #print(preds.numpy()[start:end,:].shape)
        #print(preds.numpy()[start:end,:][acquire_indices, :])
        start = end

    
def analyze_sample_efficiency(logger, tx):
    """ When we generate unlabeled samples, we do so hoping to find informative
    towers (based on the BALD acquisition function). However, it is unclear how 
    many samples are needed.

    This function will plot the acquisition function for the most informative 
    sample found vs. number of samples.
    """
    n_repeats = 10
    ensemble = logger.get_ensemble(tx)

    tower_keys = ['2block', '3block', '4block', '5block']
    max_scores = {k: [] for k in tower_keys}
    lower_scores = {k: [] for k in tower_keys}
    upper_scores = {k: [] for k in tower_keys}

    for n_samples in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
        it_scores = {k: [] for k in tower_keys}
        for _ in range(n_repeats):
            unlabeled = sample_unlabeled_data(n_samples)
            preds = get_predictions(unlabeled, ensemble)
            bald_scores = bald(preds).numpy()
            # Get the max score per tower_height.
            start = 0
            for k in tower_keys:
                end = start + unlabeled[k]['towers'].shape[0]
                acquire_indices = np.argsort(bald_scores[start:end])[::-1][:10]
                it_scores[k].append(bald_scores[start:end][acquire_indices[9]])
                start = end
        for k in tower_keys:
            max_scores[k].append(np.median(it_scores[k]))
            lower_scores[k].append(np.quantile(it_scores[k], 0.05))
            upper_scores[k].append(np.quantile(it_scores[k], 0.95))
        print('-----')
        print('Median:', max_scores)
        print('Lower:', lower_scores)
        print('Upper:', upper_scores)
    
        # with open(logger.get_figure_path('max_acquisitions.pkl'), 'wb') as handle:
        #     pickle.dump((max_scores, lower_scores, upper_scores), handle)


def plot_sample_efficiency(logger):
    with open(logger.get_figure_path('min_acquisitions.pkl'), 'rb') as handle:
        scores, lower, upper = pickle.load(handle)
    tower_keys = ['2block', '3block', '4block', '5block']
    xs = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000][:len(scores[tower_keys[0]])]
    cs = ['r', 'g', 'b', 'c']
    
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    for ix, k in enumerate(tower_keys):
        axes[ix].plot(xs, scores[k], label=k, c=cs[ix])
        axes[ix].fill_between(xs, lower[k], upper[k], color=cs[ix], alpha=0.1)
        axes[ix].set_ylim(0, 0.7)
        axes[ix].legend()
        axes[ix].set_xscale('log')
    plt.show()








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    args = parser.parse_args()
    
    logger = ActiveExperimentLogger(args.exp_path)
    logger.args.max_acquisitions = 341
    plot_sample_efficiency(logger)
    analyze_sample_efficiency(logger, 341)
    # analyze_bald_scores(logger)
    #analyze_single_dataset(logger)
    get_dataset_statistics(logger)
    accs = get_validation_accuracy(logger,
                                   'learning/data/random_blocks_(x2000)_5blocks_uniform_mass.pkl')
    
    print(accs)