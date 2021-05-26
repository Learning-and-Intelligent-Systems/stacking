import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pybullet as p
import time
import torch

from block_utils import Object, get_rotated_block, World, Environment
from learning.active.acquire import bald
from learning.active.utils import ActiveExperimentLogger
from learning.domains.towers.active_utils import get_sequential_predictions, get_predictions, sample_unlabeled_data, get_labels, sample_sequential_data
from learning.domains.towers.generate_tower_training_data import sample_random_tower
from learning.domains.towers.tower_data import TowerDataset, TowerSampler, add_placement_noise, unprocess, recover_dict_from_tower_dataset
from learning.evaluate.planner import EnsemblePlanner

from tower_planner import TowerPlanner

from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

def inspect_validation_set(fname):
    with open(fname, 'rb') as handle:
        val_towers = pickle.load(handle)

    tp = TowerPlanner(stability_mode='contains')
    
    towers = val_towers['5block']['towers']
    labels = val_towers['5block']['labels']

    # Check how may towers fall over at a lower block.
    for tower_vec, label in zip(towers, labels):
        if label == 0:
            tower = [Object.from_vector(tower_vec[bx, :]) for bx in range(0, tower_vec.shape[0])]
            print(tp.tower_is_constructable(tower[:1]))

def plot_train_accuracy(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    accs = {k: [] for k in tower_keys}

    # go through each acqisition step
    for tx in range(0, logger.args.max_acquisitions):
        print('Eval timestep, ', tx)

        # load the dataset and ensemble from that timestep
        ensemble = logger.get_ensemble(tx)
        train_dict = recover_dict_from_tower_dataset(logger.load_dataset(tx))

        # make a prediction for each item in the dataset
        if hasattr(logger.args, 'sampler') and logger.args.sampler == 'sequential':
            preds = get_sequential_predictions(train_dict, ensemble, use_latents=logger.use_latents).mean(1).numpy()
        else:
            preds = get_predictions(train_dict, ensemble, use_latents=logger.use_latents).mean(1).numpy()

        # compute accuracy for each tower size
        start = 0
        for k in tower_keys:
            end = start + train_dict[k]['towers'].shape[0]
            acc = ((preds[start:end]>0.5) == train_dict[k]['labels']).mean()
            accs[k].append(acc)
            start = end

    # now plot the accuracy for each tower height
    for k in tower_keys:
        plt.plot(accs[k], label=k)

    plt.xlabel('Acquisition Step')
    plt.ylabel('Train Accuracy')
    plt.title('Train accuracy during active learning')
    plt.legend()
    plt.savefig(logger.get_figure_path('train_accuracy.png'))
    plt.clf()


def plot_latent_uncertainty(logger):
    scales = []

    # go through each acqisition step
    for tx in range(0, logger.args.max_acquisitions):
        print('Eval timestep, ', tx)

        # load the dataset and ensemble from that timestep
        ensemble = logger.get_ensemble(tx)
        scales.append(torch.exp(ensemble.latent_logscales).mean(axis=0).detach().numpy())

    plt.plot(np.array(scales))

    plt.xlabel('Acquisition Step')
    plt.ylabel('Mean Latent Scale')
    plt.title('Variance along each latent dimension')
    plt.savefig(logger.get_figure_path('latent_scale.png'))
    plt.clf()


def get_validation_accuracy(logger, fname):
    tower_keys = ['2block', '3block', '4block', '5block']
    accs = {k: [] for k in tower_keys}
    
    with open(fname, 'rb') as handle:
        val_towers = pickle.load(handle)


    for tx in range(0, logger.args.max_acquisitions):
        if tx % 10 == 0:
            print('Eval timestep, ', tx)
        ensemble = logger.get_ensemble(tx)
        if hasattr(logger.args, 'sampler') and logger.args.sampler == 'sequential':
            preds = get_sequential_predictions(val_towers, ensemble, use_latents=logger.use_latents).mean(1).numpy()
        else:
            preds = get_predictions(val_towers, ensemble, use_latents=logger.use_latents).mean(1).numpy()

        start = 0
        for k in tower_keys:
            end = start + val_towers[k]['towers'].shape[0]
            print(val_towers[k]['labels'])
            acc = ((preds[start:end]>0.5) == val_towers[k]['labels']).mean()
            f1 = f1_score(val_towers[k]['labels'], preds[start:end] > 0.5)
            print('Acc:', tx, k, acc)
            accs[k].append(acc)
            mask = (preds[start:end] > 0.9) | (preds[start:end] < 0.1)
            #print(mask)
            conf_preds = preds[start:end][mask]
            #print(conf_preds)
            conf_labels = val_towers[k]['labels'][mask]
            conf = ((conf_preds>0.5) == conf_labels)
            #print(k, conf.mean())
            start = end
        
    
    with open(logger.get_figure_path('val_accuracies.pkl'), 'wb') as handle:
        pickle.dump(accs, handle)
    return accs

def plot_val_accuracy(logger, init_dataset_size=100, n_acquire=10, separate_axs=False):
    with open(logger.get_figure_path('val_accuracies.pkl'), 'rb') as handle:
        accs = pickle.load(handle)

    tower_keys = ['2block', '3block', '4block', '5block']

    ref_acc = {'2block': .955, '3block': .925, '4block': .912, '5block': .913}
    

    if separate_axs:
        plt.clf()
        fig, axes = plt.subplots(4, figsize=(10, 20))
        for ix, ax in enumerate(axes):
            k = tower_keys[ix]
            max_x = init_dataset_size + n_acquire*len(accs[k])
            xs = np.arange(init_dataset_size, max_x, n_acquire)

            ax.plot(xs, accs[k], label=k)
            #ax.plot([400, 4500], [ref_acc[k]]*2)
            #ax.axvline(x=4375)
            ax.set_xlabel('Number of Towers')
            ax.set_ylabel('Val Accuracy')
            ax.legend()
    else:
        for k in tower_keys:
            max_x = init_dataset_size + n_acquire*len(accs[k])
            xs = np.arange(init_dataset_size, max_x, n_acquire)
            plt.plot(xs, accs[k], label=k)
        plt.xlabel('Number of Towers')
        plt.ylabel('Val Accuracy')
        plt.legend()

    plt.title('Validation Accuracy throughout active learning')
    plt.savefig(logger.get_figure_path('val_accuracy.png'))
    plt.clf()

def get_dataset_statistics(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    aq_over_time = np.zeros((logger.args.max_acquisitions, 4))

    for tx in range(logger.args.max_acquisitions):
        acquired_data, _ = logger.load_acquisition_data(tx)
        
        nums = []
        pos = []
        for kx, key in enumerate(tower_keys):
            if key in acquired_data:
                nums += [acquired_data[key]['towers'].shape[0]]
                pos += [np.sum(acquired_data[key]['labels'])]
        #nums = [acquired_data[k]['towers'].shape[0] for k in tower_keys]
        #pos = [np.sum(acquired_data[k]['labels']) for k in tower_keys]
                print(nums, pos)

        
                aq_over_time[tx, kx] = acquired_data[key]['towers'].shape[0]

    

    print('Totals')
    for tx in range(logger.args.max_acquisitions):
        dataset = logger.load_dataset(tx)
        print([dataset.tower_tensors[k].shape[0] for k in tower_keys])

    plt.clf()
    bars = plt.bar([2, 3, 4, 5], [dataset.tower_tensors[k].shape[0]/4 for k in tower_keys])
    plt.xticks([2, 3, 4, 5], ['2 Blocks', '3 Blocks', '4 Blocks', '5 Blocks'])
    bars[0].set_color('b')
    bars[1].set_color('tab:orange')
    bars[2].set_color('g')
    bars[3].set_color('r')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.ylabel('Number of Towers', fontsize=20)
    plt.show()

    max_x = 40 + 10*logger.args.max_acquisitions
    xs = np.arange(40, max_x, 10)
    print(xs.shape)
    xs = np.arange(1, logger.args.max_acquisitions+1)
    print(xs.shape, aq_over_time.shape)
    print(xs)

    w = 1
    plt.figure(figsize=(20, 10))
    plt.bar(xs, aq_over_time[:, 0], width=w, label='2block')
    for kx in range(1, len(tower_keys)):
        plt.bar(xs, aq_over_time[:, kx], bottom=np.sum(aq_over_time[:, :kx], axis=1), width=w, label=tower_keys[kx])
    
    plt.xlabel('Acquisition Step', fontsize=20)
    plt.ylabel('Number of Collected Samples', fontsize=20)
    plt.legend(prop={'size': 20})
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(logger.get_figure_path('acquisition_breakdown.png'))

def plot_constructability_over_time(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    tallest_stable_over_time = np.zeros((logger.args.max_acquisitions, 5))
    tp = TowerPlanner(stability_mode='contains')
    for tx in range(logger.args.max_acquisitions):
        acquired_data, _ = logger.load_acquisition_data(tx)

        # For each tower, figure out when it fell over.
        for kx, k in enumerate(tower_keys):
            towers = acquired_data[k]['towers']
            for ix in range(0, towers.shape[0]):
                height = 1
                for top_id in range(1, towers.shape[1]):
                    block_tower = [Object.from_vector(towers[ix, bx, :]) for bx in range(0, top_id+1)]
                    if not tp.tower_is_constructable(block_tower):
                        break
                    height += 1
                tallest_stable_over_time[tx, height-1] += 1

    max_x = 40 + 10*logger.args.max_acquisitions
    xs = np.arange(40, max_x, 10)

    w = 10
    plt.figure(figsize=(20, 10))
    plt.bar(xs, tallest_stable_over_time[:, 0], width=w, label=1)
    for kx in range(1, 5):
        plt.bar(xs, tallest_stable_over_time[:, kx], bottom=np.sum(tallest_stable_over_time[:, :kx], axis=1), width=w, label=kx+1)
    
    plt.xlabel('Acquisition Step')
    plt.ylabel('Height of tallest stable subtower')
    plt.legend()
    plt.savefig(logger.get_figure_path('tallest_breakdown.png'))

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

    tx = 10
    acquired, unlabeled = logger.load_acquisition_data(tx)
    ensemble = logger.get_ensemble(tx)
    preds = get_predictions(unlabeled, ensemble, use_latents=logger.use_latents)
    bald_scores = bald(preds).numpy()
    acquire_indices = np.argsort(bald_scores)[::-1][:10]
    start = 0
    for k in tower_keys:
        end = start + unlabeled[k]['towers'].shape[0]
        acquire_indices = np.argsort(bald_scores[start:end])[::-1][:25]
        print(bald_scores[start:end][acquire_indices])
        print(acquire_indices + start)
        #print(preds.numpy()[start:end,:].shape)
        #print(preds.numpy()[start:end,:][acquire_indices, :])
        start = end

def get_acquisition_scores_over_time(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    scores = {k: [] for k in tower_keys}

    for tx in range(0, logger.args.max_acquisitions):
        if tx % 10 == 0:
            print(tx)
        _, unlabeled = logger.load_acquisition_data(tx)
        ensemble = logger.get_ensemble(tx)

        preds = get_predictions(unlabeled, ensemble, use_latents=logger.use_latents)
        bald_scores = bald(preds).numpy()
        # Get the max score per tower_height.
        start = 0
        for k in tower_keys:
            end = start + unlabeled[k]['towers'].shape[0]
            acquire_indices = np.argsort(bald_scores[start:end])[::-1][:10]
            scores[k].append(bald_scores[start:end][acquire_indices[0]])
            start = end

        with open(logger.get_figure_path('acquisition_over_time.pkl'), 'wb') as handle:
            pickle.dump(scores, handle)

def plot_acquisition_scores_over_time(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    with open(logger.get_figure_path('acquisition_over_time.pkl'), 'rb') as handle:
        scores = pickle.load(handle)
    xs = np.arange(0, len(scores[tower_keys[0]]))
    plt.figure(figsize=(20,5))
    for k in tower_keys:
        plt.plot(xs, scores[k], label=k)
    plt.legend()
    plt.ylabel('Maximum BALD Score')
    plt.xlabel('Acquisition Step')
    plt.savefig(logger.get_figure_path('acquisition_over_time.png'))


def analyze_acquisition_histogram(logger, block_set_fname='learning/data/block_set_10.pkl'):
    """ When we generate unlabeled samples, we do so hoping to find informative
    towers (based on the BALD acquisition function). However, it is unclear how 
    many samples are needed.

    This function will plot the acquisition function for the most informative 
    sample found vs. number of samples.
    """
    n_repeats = 1
    tower_keys = ['2block', '3block', '4block', '5block']


    overall_data = {}
    
    for n_samples in [100000]:
        with open(block_set_fname, 'rb') as handle:
            block_set = pickle.load(handle)

        unlabeled_data = [sample_unlabeled_data(n_samples, block_set) for _ in range(n_repeats)]
        
        # noisy_data = copy.deepcopy(unlabeled_data)
        # for unlabeled in noisy_data:
        #     for k in tower_keys:
        #         unlabeled[k]['towers'] = add_placement_noise(unlabeled[k]['towers'])
        
        for tx in range(200, 250, 1):
            print(tx)
            dataset = logger.load_dataset(tx)
            #unlabeled_data = [sample_sequential_data(block_set, dataset, n_samples) for _ in range(n_repeats)]

            ensemble = logger.get_ensemble(tx)
            it_scores = {k: [] for k in tower_keys}
            for ux in range(n_repeats):
                unlabeled = unlabeled_data[ux]
                #noisy = noisy_data[ux]
                # with open('learning/data/random_blocks_(x40000)_5blocks_uniform_mass.pkl', 'rb') as handle:
                #     unlabeled = pickle.load(handle)
                preds = get_predictions(unlabeled, ensemble, use_latents=logger.use_latents)[:,:]
                #noisy_preds = get_predictions(noisy, ensemble)
                labels = get_labels(unlabeled)
                bald_scores = bald(preds).numpy()

                

                #noisy_bald_scores = bald(noisy_preds).numpy()

                # Get the max score per tower_height.
                start = 0
                fig, axes = plt.subplots(4, figsize=(10, 20), sharex=True)
                for kx, k in enumerate(tower_keys):

                    print(k)
                    end = start + unlabeled[k]['towers'].shape[0]
                    acquire_indices = np.argsort(bald_scores[start:end])[::-1][:100]
                    

                    # Print information about how many non-informative towers are wrong.
                    #low_indices = np.argsort(bald_scores[start:end])[:]
                    low_indices = (bald_scores[start:end] > 0.4).nonzero()[0]

                    masses = unlabeled[k]['towers'][low_indices, :, 0]
                    block_sets = set([tuple(masses[ix, :].tolist()) for ix in range(masses.shape[0])])
                    print(low_indices.shape[0], len(block_sets))
                    print(bald_scores.shape)
                    ls = labels[k]['labels'][low_indices]
                    ps = preds.numpy()[start:end][low_indices]
                    bs = bald_scores[start:end][low_indices]

                    acc = ((ps.mean(1) > 0.5) == ls).mean()
                    
                    print(acc)
                    # for lx in range(0, low_indices.shape[0]):
                    #     print(ls[-lx], ps[-lx,:].mean(), ps[-lx,:], bs[-lx])

                    # Print information about top scoring towers.
                    # print(k) 
                    # print(labels[k]['labels'][acquire_indices[0:10]])
                    # print(preds.numpy()[start:end][acquire_indices[0:10],:])
                    # print(bald_scores[start:end][acquire_indices[0:10]])

                    # Check acquire indices to see if we already have that tower.
                    block_orders = set()
                    pruned_indices = []
                    for ix in acquire_indices:
                        masses = tuple(unlabeled[k]['towers'][ix, :, 0].tolist())
                        if masses not in block_orders:
                            pruned_indices.append(ix)
                            block_orders.add(tuple(masses))
                    print(len(acquire_indices), len(pruned_indices))


                    it_scores[k].append(bald_scores[start:end][acquire_indices[0]])
                    #it_scores[k].append(np.mean(bald_scores[start:end][acquire_indices]))
                    axes[kx].hist(bald_scores[start:end][pruned_indices], bins=50)
                    #axes[kx].hist(preds.mean(1)[start:end], bins=50)
                    start = end
                    axes[kx].set_xlim(0, 0.7)
       
                plt.show()


def analyze_acquisition_value_with_sampling_size(logger):
    """ When we generate unlabeled samples, we do so hoping to find informative
    towers (based on the BALD acquisition function). However, it is unclear how 
    many samples are needed.

    This function will plot the acquisition function for the most informative 
    sample found vs. number of samples.
    """
    n_repeats = 10
    tower_keys = ['2block', '3block', '4block', '5block']


    overall_data = {}
    
    for n_samples in [100, 1000, 10000]:
        overall_data[n_samples] = {}
        overall_data[n_samples]['median'] = {k: [] for k in tower_keys}
        overall_data[n_samples]['lower'] = {k: [] for k in tower_keys}
        overall_data[n_samples]['upper'] = {k: [] for k in tower_keys}
        
        unlabeled_data = [sample_unlabeled_data(n_samples) for _ in range(n_repeats)]

        for tx in range(0, logger.args.max_acquisitions):
            print(tx)
            ensemble = logger.get_ensemble(tx)
            it_scores = {k: [] for k in tower_keys}
            for ux in range(n_repeats):
                unlabeled = unlabeled_data[ux]

                preds = get_predictions(unlabeled, ensemble, use_latents=logger.use_latents)[:,:]
                bald_scores = bald(preds).numpy()
                # Get the max score per tower_height.
                start = 0
                for k in tower_keys:
                    end = start + unlabeled[k]['towers'].shape[0]
                    acquire_indices = np.argsort(bald_scores[start:end])[::-1][:10]
                    #it_scores[k].append(bald_scores[start:end][acquire_indices[0]])
                    it_scores[k].append(np.mean(bald_scores[start:end][acquire_indices]))
                    start = end
            for k in tower_keys:
                overall_data[n_samples]['median'][k].append(np.median(it_scores[k]))
                overall_data[n_samples]['lower'][k].append(np.quantile(it_scores[k], 0.05))
                overall_data[n_samples]['upper'][k].append(np.quantile(it_scores[k], 0.95))

    
            with open(logger.get_figure_path('time_acquisitions_mean.pkl'), 'wb') as handle:
                pickle.dump(overall_data, handle)

def plot_acquisition_value_with_sampling_size(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    with open(logger.get_figure_path('time_acquisitions_mean.pkl'), 'rb') as handle:
        overall_data = pickle.load(handle)
    
    fig, axes = plt.subplots(4, figsize=(10, 16))
    for kx in range(0, len(tower_keys)):
        k = tower_keys[kx]
        for n_samples in overall_data.keys():
            med = overall_data[n_samples]['median'][k]
            low = overall_data[n_samples]['lower'][k]
            high = overall_data[n_samples]['upper'][k]
            axes[kx].set_ylabel('Max BALD Score')
            axes[kx].set_xlabel('Acquisition Step')
            axes[kx].plot(np.arange(0, len(med)), med, label=n_samples)
            axes[kx].fill_between(np.arange(0, len(med)), low, high, alpha=0.2)
            axes[kx].set_ylim(0, 0.5)
            axes[kx].legend()
    plt.savefig(logger.get_figure_path('mean_bald_vs_sample_size.png'))

    
def analyze_sample_efficiency(logger, tx):
    """ When we generate unlabeled samples, we do so hoping to find informative
    towers (based on the BALD acquisition function). However, it is unclear how 
    many samples are needed.

    This function will plot the acquisition function for the most informative 
    sample found vs. number of samples.
    """
    n_repeats = 50
    ensemble = logger.get_ensemble(tx)

    tower_keys = ['2block', '3block', '4block', '5block']
    max_scores = {k: [] for k in tower_keys}
    lower_scores = {k: [] for k in tower_keys}
    upper_scores = {k: [] for k in tower_keys}

    for n_samples in [1000]:#[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
        it_scores = {k: [] for k in tower_keys}
        for _ in range(n_repeats):
            unlabeled = sample_unlabeled_data(n_samples)
            # TODO: Remove this later.
            # for k in tower_keys[:3]:
            #     unlabeled[k]['towers'] = unlabeled['5block']['towers'][:1000,...].copy()
            #     unlabeled[k]['labels'] = unlabeled['5block']['labels'][:1000,...].copy()
            # for k in tower_keys:
            #     print(k, unlabeled[k]['towers'].shape)
            preds = get_predictions(unlabeled, ensemble, use_latents=logger.use_latents)
            bald_scores = bald(preds).numpy()

            # Get the max score per tower_height.
            start = 0
            for k in tower_keys:
                end = start + unlabeled[k]['towers'].shape[0]
                acquire_indices = np.argsort(bald_scores[start:end])[::-1][:10]
                #print(preds[start:end,...][acquire_indices[0]])
                
                max_ix = acquire_indices[0]
                pred = preds[start:end][max_ix:(max_ix+1),:]
                print(bald(pred, show=True))


                it_scores[k].append(bald_scores[start:end][acquire_indices[0]])
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


def decision_distance(tower):
    bot_dim_x, bot_dim_y = tower[0, [4, 5]]/2.
    top_com_rel_x, top_com_rel_y = tower[1, [1, 2]]
    top_pos_x, top_pos_y = tower[1, [7, 8]]

    top_com_x = top_pos_x + top_com_rel_x
    top_com_y = top_pos_y + top_com_rel_y
    x_dist = np.abs(np.abs(top_com_x) - bot_dim_x)
    y_dist = np.abs(np.abs(top_com_y) - bot_dim_y)
    return np.min([x_dist, y_dist])

def analyze_collected_2block_towers(logger):

    distances = []
    for tx in range(logger.args.max_acquisitions):
        towers, _ = logger.load_acquisition_data(tx)
        print('-----')
        dist = []
        for tower in towers['2block']['towers']:
            # TODO: Check how close to the boundary the tower is.
            d = decision_distance(tower)
            dist.append(d)
        distances.append(np.mean(dist))
        #print(top_com_x/bot_dim_x, top_com_y/bot_dim_y)
    plt.plot(np.arange(0, len(distances)), distances)
    plt.xlabel('Acquisition Step')
    plt.ylabel('Distance to Edge')
    plt.savefig(logger.get_figure_path('2block_analysis.png'))

        #print(tower[0,[1,2,4,5,7,8]], tower[1,[1,2,4,5,7,8]])

def inspect_2block_towers(logger):
    """
    In the full dataset, show the distribution of features.
    """
    tower_keys = ['2block', '3block', '4block', '5block']

    # dataset = logger.load_dataset(logger.args.max_acquisitions - 1)
    # print(dataset.tower_tensors['2block'].shape)
    # plt.hist(dataset.tower_tensors['2block'][:,1,8], bins=10)
    # plt.show()

    #ensemble = logger.get_ensemble(10)
    ensemble = logger.get_ensemble(logger.args.max_acquisitions - 1)
    unlabeled = sample_unlabeled_data(10000)
    preds = get_predictions(unlabeled, ensemble, use_latents=logger.use_latents)
    bald_scores = bald(preds).numpy()
    print('Best BALD')
    ixs = np.argsort(bald_scores)[::-1][:10]
    print(bald_scores[ixs])
    input()
    tp = TowerPlanner(stability_mode='contains')
    preds2 = preds[:unlabeled['2block']['towers'].shape[0], :]
    bald_scores2 = bald_scores[:unlabeled['2block']['towers'].shape[0]]
    acquire_indices = np.argsort(bald_scores2)[::-1][:50]
    # for ix in range(preds2.shape[0]):
    #     print(np.around(preds2[ix,:].numpy(), 2), np.around(bald_scores2[ix], 3))
    print('-----')
    for ix in acquire_indices:
        d = decision_distance(unlabeled['2block']['towers'][ix,:,:])
        tower = unlabeled['2block']['towers'][ix,:,:]
        l = tp.tower_is_constructable([Object.from_vector(tower[bx, :]) for bx in range(tower.shape[0])])
        print(np.around(preds2[ix,:].numpy(), 4), np.around(bald_scores2[ix], 3), d, l)

    for ix in acquire_indices:
        unlabeled['2block']['towers'][ix,1,7:8] += 0.0
    new_preds = get_predictions(unlabeled, ensemble, use_latents=logger.use_latents)
    print('-----')
    for ix in acquire_indices:
        d = decision_distance(unlabeled['2block']['towers'][ix,:,:])
        print(np.around(new_preds[ix,:].numpy(), 2))
    plt.hist(unlabeled['2block']['towers'][acquire_indices,1,0])
    plt.show()
    print('-----')
    start = 0
    for k in tower_keys:
        end = start + unlabeled[k]['towers'].shape[0]
        p, b = preds[start:end, :], bald_scores[start:end]
        informative = b[b > 0.3]
        print(p.shape, informative.shape)

    accs = {k: [] for k in tower_keys}
    with open('learning/data/random_blocks_(x2000)_5blocks_uniform_mass.pkl', 'rb') as handle:
        val_towers = pickle.load(handle)

    preds = get_predictions(val_towers, ensemble, use_latents=logger.use_latents).mean(1).numpy()
    dists = []
    for ix in range(0, val_towers['2block']['towers'].shape[0]):
        d = decision_distance(val_towers['2block']['towers'][ix,:,:])
        dists.append(d)
    print(len(dists))
    plt.hist(dists, bins=100)
    plt.show()

    start = 0
    for k in tower_keys:
        end = start + val_towers[k]['towers'].shape[0]
        acc = ((preds[start:end]>0.5) == val_towers[k]['labels']).mean()
        accs[k].append(acc)
        start = end
    print(accs)

def plot_2block_BALD_heatmap(logger):
    # TODO: Find informative 2-block tower.

    # TODO: Grid the x-y position of the top block.

    # TODO: Plot the predictions overlayed on the top block.
    pass


def single_2block_tower_analysis(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    ensemble = logger.get_ensemble(logger.args.max_acquisitions - 1)
    unlabeled = sample_unlabeled_data(1000)
    tower = unlabeled['2block']['towers'][0:1,:,:].copy()
    displacements = np.linspace(-0.1, 0.1, 1000).reshape(1000,1,1)
    unlabeled['2block']['towers'] = np.resize(tower, (1000, 2, 21))
    unlabeled['2block']['labels'] = np.zeros((1000,))
    print(displacements.shape, unlabeled['2block']['towers'][:,1,7:8].shape)
    unlabeled['2block']['towers'][:,1:2,7:8] += displacements

    preds = get_predictions(unlabeled, ensemble, use_latents=logger.use_latents)[:1000,...]

    for ix in range(1000):
        dim_x = unlabeled['2block']['towers'][ix, 0, 4]/2.
        com_x = unlabeled['2block']['towers'][ix, 1, 1] + unlabeled['2block']['towers'][ix, 1, 7]
        print(np.around(preds[ix,:], 3), dim_x, com_x)
    dim_y = unlabeled['2block']['towers'][ix, 0, 5]/2.
    com_y = unlabeled['2block']['towers'][ix, 1, 2] + unlabeled['2block']['towers'][ix, 1, 8]
    print(dim_y, com_y)
    bald_scores = bald(preds).numpy()


def check_validation_robustness(noise=0.001, n_attempts=10):
    """
    Try adding noise to the placement of each block in the validation set
    to see how many of the towers are robust to placement noise.
    """
    with open('learning/data/validation_towers_robust.pkl', 'rb') as handle:
    #with open('learning/data/random_blocks_(x2000)_5blocks_uniform_mass.pkl', 'rb') as handle:
        val_towers = pickle.load(handle)
    robust = {k: 0 for k in val_towers.keys()}
    tp = TowerPlanner(stability_mode='contains')
    stable_towers = copy.deepcopy(val_towers)
    unstable_towers = copy.deepcopy(val_towers)
    for k in robust.keys():
        stable_indices = []
        unstable_indices = []
        for ix in range(0, val_towers[k]['towers'].shape[0]):
            stable = True
            if val_towers[k]['labels'][ix] == 0:
                continue

            for _ in range(n_attempts):
                tower = val_towers[k]['towers'][ix, :, :].copy()
                label = val_towers[k]['labels'][ix]
                tower[:, 7:9] += np.random.randn(2*tower.shape[0]).reshape(tower.shape[0], 2)*noise

                block_tower = [Object.from_vector(tower[kx, :]) for kx in range(tower.shape[0])]

                if tp.tower_is_constructable(block_tower) != label:
                    stable = False

            if stable:
                robust[k] += 1
                stable_indices.append(ix)
            else:
                unstable_indices.append(ix)
        
        stable_towers[k]['towers'] = stable_towers[k]['towers'][stable_indices,...]
        stable_towers[k]['labels'] = stable_towers[k]['labels'][stable_indices]
        
        unstable_towers[k]['towers'] = unstable_towers[k]['towers'][unstable_indices,...]
        unstable_towers[k]['labels'] = unstable_towers[k]['labels'][unstable_indices]
        
        # with open('learning/data/stable_val.pkl', 'wb') as handle:
        #     pickle.dump(stable_towers, handle)

        # with open('learning/data/unstable_val.pkl', 'wb') as handle:
        #     pickle.dump(unstable_towers, handle)
        
        print(k, ':', robust[k], '/', val_towers[k]['towers'].shape[0] )


def tallest_tower_regret_evaluation(logger, block_set, fname, args, save_imgs=False):
    def tower_height(tower):
        """
        :param tower: A vectorized version of the tower.
        """
        return np.sum(tower[:, 6])

    return evaluate_planner(logger, block_set, tower_height, fname, args, save_imgs, img_prefix='height')

def cumulative_overhang_regret_evaluation(logger, block_set, fname, args, save_imgs=False):
    def horizontal_overhang(tower):
        total_overhang = 0
        for tx in range(1, tower.shape[0]):
            bx = tx - 1
            overhang = (tower[tx, 7] + tower[tx, 4]/2.) - (tower[bx, 7] + tower[bx, 4]/2.)
            total_overhang += overhang
        return total_overhang
    
    return evaluate_planner(logger, block_set, horizontal_overhang, fname, args, save_imgs, img_prefix='cumulative_overhang')
    
def longest_overhang_regret_evaluation(logger, block_set, fname, args, save_imgs=False):
    def horizontal_overhang(tower):
        return (tower[-1, 7] + tower[-1, 4]/2.) - (tower[0, 7] + tower[0, 4]/2.)
    
    return evaluate_planner(logger, block_set, horizontal_overhang, fname, args, save_imgs, img_prefix='overhang')
    
def min_contact_regret_evaluation(logger, block_set, fname, args, save_imgs=False):
    def contact_area(tower):
        """
        :param tower: A vectorized version of the tower.
        """
        lefts, rights = tower[:, 7] - tower[:, 4]/2., tower[:, 7] + tower[:, 4]/2.
        bottoms, tops = tower[:, 8] - tower[:, 5]/2., tower[:, 8] + tower[:, 5]/2.

        area = 0.
        for tx in range(1, tower.shape[0]):
            bx = tx - 1
            l, r = max(lefts[bx], lefts[tx]), min(rights[bx], rights[tx])
            b, t = max(bottoms[bx], bottoms[tx]), min(tops[bx], tops[tx])
            
            top_area = tower[tx, 4]*tower[tx, 5]
            contact = (r-l)*(t-b)
            area += top_area - contact

        return area

    return evaluate_planner(logger, block_set, contact_area, fname, args, save_imgs, img_prefix='contact')

def evaluate_planner(logger, blocks, reward_fn, fname, args, save_imgs=False, img_prefix=''):
    tower_keys = [str(ts)+'block' for ts in args.tower_sizes]
    tp = TowerPlanner(stability_mode='contains')
    ep = EnsemblePlanner(logger)

    # Store regret for towers of each size.
    regrets = {k: [] for k in tower_keys}
    rewards = {k: [] for k in tower_keys}

    if args.max_acquisitions is not None: 
        eval_range = range(0, args.max_acquisitions, 10)
    elif args.acquisition_step is not None: 
        eval_range = [args.acquisition_step]
    
    for tx in eval_range:
        print('Acquisition step:', tx)

        ensemble = logger.get_ensemble(tx)
        if torch.cuda.is_available():
            ensemble = ensemble.cuda()
            
        for k, size in zip(tower_keys, args.tower_sizes):
            print('Tower size', k)
            num_failures, num_pw_failures = 0, 0
            curr_regrets = []
            curr_rewards = []
            for t in range(0, args.n_towers):
                print('Tower number', t)
                
                if blocks is not None:
                    plan_blocks = np.random.choice(blocks, size, replace=False)	
                    plan_blocks = copy.deepcopy(plan_blocks)	
                else:
                    plan_blocks = [Object.random() for _ in range(size)]
                    
                tower, reward, max_reward, tower_block_ids = ep.plan(plan_blocks, 
                                                                ensemble, 
                                                                reward_fn,
                                                                args,
                                                                num_blocks=size,
                                                                n_tower=t)
                                
                block_tower = []
                for vec_block, block_id in zip(tower, tower_block_ids):
                    block = Object.from_vector(vec_block)
                    block.name = 'obj_%d' % block_id
                    block_tower.append(block)         
                                           
                # save tower info to /evaluation_towers
                if args.exec_mode is None:
                    if args.planning_model == 'noisy-model':
                        logger.save_evaluation_tower(block_tower, reward, max_reward, tx, args.planning_model, args.problem, noise=args.plan_xy_noise)
                    else:
                        logger.save_evaluation_tower(block_tower, reward, max_reward, tx, args.planning_model, args.problem)

                # perturb tower if evaluating with noisy model
                if args.exec_mode == 'noisy-model':
                    block_tower = []
                    for vec_block, block_id in zip(tower, tower_block_ids):
                        vec_block[7:9] += np.random.randn(2)*args.exec_xy_noise
                        block = Object.from_vector(vec_block)
                        block.name = 'obj_%d' % block_id
                        block_tower.append(block)     
    
                # build found tower
                if args.exec_mode == 'noisy-model' or args.exec_mode == 'simple-model':
                    if not tp.tower_is_constructable(block_tower):
                        reward = 0
                        num_failures += 1
                        if tp.tower_is_pairwise_stable(block_tower):
                            num_pw_failures += 1
                        else:
                            pairs = []
                            dists = []
                            for i in range(len(tower) - 1):
                                # check that each pair of blocks is stably individually
                                top = block_tower[i+1]
                                bottom = block_tower[i]
                                if not tp.pair_is_stable(bottom, top): 
                                    pairs.append(False)
                                else:
                                    pairs.append(True)
                                top_rel_pos = np.array(top.pose.pos) - np.array(bottom.pose.pos)
                                top_rel_com = top_rel_pos + top.com
                                dists.append((np.abs(top_rel_com)*2 - bottom.dimensions)[:2])
                            #print('Pairs:', pairs, dists)
                            
                    #print('PW Stable:', tp.tower_is_pairwise_stable(block_tower))
                    #print('Global Stable:', tp.tower_is_stable(block_tower))
                    
                    if False and reward != 0:
                        print(reward, max_reward)
                        w = World(block_tower)
                        env = Environment([w], vis_sim=True, vis_frames=True)
                        input()
                        for tx in range(240):
                            env.step(vis_frames=True)
                            time.sleep(1/240.)
                        env.disconnect()
                
                    # Note that in general max reward may not be the best possible due to sampling.
                    #ground_truth = np.sum([np.max(b.dimensions) for b in blocks])
                    #print(max_reward, ground_truth)

                    # Compare heights and calculate regret.
                    regret = (max_reward - reward)/max_reward
                    #print(reward, max_reward)
                    #print(regret)
                    curr_regrets.append(regret)
                    curr_rewards.append(reward)

            if args.exec_mode == 'noisy-model' or args.exec_mode == 'simple-model':
                regrets[k].append(curr_regrets)
                rewards[k].append(curr_rewards)

        if args.max_acquisitions is not None:
            if args.exec_mode == 'noisy-model' or args.exec_mode == 'simple-model':
                with open(logger.get_figure_path(fname+'_regrets.pkl'), 'wb') as handle:
                    pickle.dump(regrets, handle)
                    
                with open(logger.get_figure_path(fname+'_rewards.pkl'), 'wb') as handle:
                    pickle.dump(rewards, handle)
            
    # if just ran for one acquisition step, output final regret and reward
    if args.acquisition_step is not None:
        if args.exec_mode == 'noisy-model' or args.exec_mode == 'simple-model':
            final_median_regret = np.median(regrets[k][0])
            final_upper75_regret = np.quantile(regrets[k][0], 0.75)
            final_lower25_regret = np.quantile(regrets[k][0][0], 0.25)
            
            final_median_reward = np.median(rewards[k][0])
            final_upper75_reward = np.quantile(rewards[k][0], 0.75)
            final_lower25_reward = np.quantile(rewards[k][0], 0.25)
            
            final_average_regret = np.average(regrets[k][0])
            final_std_regret = np.std(regrets[k][0])
            
            final_average_reward = np.average(rewards[k][0])
            final_std_reward = np.std(rewards[k][0])
            
            print('Final Median Regret: (%f) %f (%f)' % (final_lower25_regret, final_median_regret, final_upper75_regret))
            print('Final Median Reward: (%f) %f (%f)' % (final_lower25_reward, final_median_reward, final_upper75_reward))
            
            print('Final Average Regret: %f +/- %f' % (final_average_regret, final_std_regret))
            print('Final Average Reward: %f +/- %f' % (final_average_reward, final_std_reward))

def plot_regret(logger, fname):
    with open(logger.get_figure_path(fname), 'rb') as handle:
        regrets = pickle.load(handle)

    tower_keys = ['2block', '3block', '4block', '5block']
    upper975 = {k: [] for k in tower_keys}
    upper75 = {k: [] for k in tower_keys}
    median = {k: [] for k in tower_keys}
    lower25 = {k: [] for k in tower_keys}
    lower025 = {k: [] for k in tower_keys}
    for k in tower_keys:
        rs = regrets[k]
        #print(rs)
        for tx in range(len(rs)):
            median[k].append(np.median(rs[tx]))
            lower025[k].append(np.quantile(rs[tx], 0.05))
            lower25[k].append(np.quantile(rs[tx], 0.25))
            upper75[k].append(np.quantile(rs[tx], 0.75))
            upper975[k].append(np.quantile(rs[tx], 0.95))
    fig, axes = plt.subplots(4, sharex=True, figsize=(10,20))
    for kx, k in enumerate(tower_keys):
        #xs = np.arange(400, 400+100*len(median[k]), 100)
        xs = np.arange(40, 40+100*len(median[k]), 100)
        #xs = np.arange(40, 40+10*len(median[k]), 10)
        axes[kx].plot(xs, median[k], label=k)
        axes[kx].fill_between(xs, lower25[k], upper75[k], alpha=0.2)
        #axes[kx].fill_between(xs, lower025[k], upper975[k], alpha=0.2)
        axes[kx].set_ylim(0.0, 1.1)
        axes[kx].set_ylabel('Regret (Normalized)')
        axes[kx].set_xlabel('Number of training towers')
        axes[kx].legend()
    plt_fname = fname[:-4]+'.png'
    plt.savefig(logger.get_figure_path(plt_fname))

def get_percent_stable(logger):
    dataset = logger.load_dataset(100)
    tower_keys = ['2block', '3block', '4block', '5block']

    for k in tower_keys:
        print(k, dataset.tower_labels[k].mean())


def save_collected_tower_images(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    for tx in range(0, 20):
        towers, _ = logger.load_acquisition_data(tx)
        
        ix = np.random.randint(0, 10)
        print(towers['2block']['towers'].shape)
        start = 0 
        for k in tower_keys:
            end = start + towers[k]['towers'].shape[0]

            if ix < end:
                tower = towers[k]['towers'][ix - start, :, :]
                block_tower = [Object.from_vector(tower[bx, :]) for bx in range(tower.shape[0])]
                label = towers[k]['labels'][ix-start]
                break
            start = end

        w = World(block_tower)
        env = Environment([w], vis_sim=False, vis_frames=True)
        env.step(vis_frames=True)

        view_matrix = p.computeViewMatrixFromYawPitchRoll(distance=0.3,
                                                            yaw=45,
                                                            pitch=-10,
                                                            roll=0,
                                                            upAxisIndex=2,
                                                            cameraTargetPosition=(0., 0., 0.25))
        aspect = 100. / 190.
        nearPlane = 0.01
        farPlane = 10
        fov = 90
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        image_data = p.getCameraImage(200, 380, shadow=1,  viewMatrix=view_matrix, projectionMatrix=projection_matrix)
        w, h, im = image_data[:3]
        np_im = np.array(im, dtype=np.uint8).reshape(h, w, 4)[:, :, 0:3]
        #plt.imshow(np.array(np_im))
        plt.imsave(logger.get_figure_path('tower_%d_%d.png' % (tx, label)), np_im)
        env.disconnect()

def check_redundancy(logger):
    for tx in range(0, logger.args.max_acquisitions):
        towers, _ = logger.load_acquisition_data(tx)

        # Check acquire indices to see if we already have that tower.
        
        for k in ['2block']:#towers.keys():
            print(tx, k)
            block_orders = set()
            pruned_indices = []
            for ix in range(0, towers[k]['towers'].shape[0]):
                masses = tuple(towers[k]['towers'][ix, :, 0].tolist())
                if masses not in block_orders:
                    pruned_indices.append(ix)
                    block_orders.add(masses)
            print(towers[k]['towers'].shape[0], len(pruned_indices))

def check_stable_bases(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    tp = TowerPlanner(stability_mode='contains')
    for tx in range(0, 80):
        print(tx)
        towers, _ = logger.load_acquisition_data(tx)

        for k in tower_keys:
            print(k)
            for tower_vec in towers[k]['towers']:
                tower = [Object.from_vector(tower_vec[bx, :]) for bx in range(0, tower_vec.shape[0])]
                if tp.tower_is_constructable(tower[:-1]):
                    print('Stable Base')
                else:
                    print('Unstable Base')
                    #print(tower_vec)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    args = parser.parse_args()
    
    logger = ActiveExperimentLogger(args.exp_path, use_latents=True)
    logger.args.max_acquisitions = 50

    plot_train_accuracy(logger)
    plot_latent_uncertainty(logger)


    #plot_sample_efficiency(logger)
    #analyze_sample_efficiency(logger, 340)
    # analyze_bald_scores(logger)
    # get_acquisition_scores_over_time(logger)
    # plot_acquisition_scores_over_time(logger)
    #analyze_single_dataset(logger)
    
    #check_stable_bases(logger)
    # get_dataset_statistics(logger)

    #inspect_validation_set('learning/data/1000block_set_(x1000.0)_constructable__val_10block.pkl')

    accs = get_validation_accuracy(logger,
                                  # 'learning/data/1000block_set_(x1000.0)_constructable__val_10block.pkl')
                                  '/Users/izzy/projects/stacking/learning/data/may_cubes/towers/10block_set_(x1000)_seq_a_2_dict.pkl')
    plot_val_accuracy(logger)

    # #analyze_collected_2block_towers(logger)
    # print(accs)

    #analyze_acquisition_value_with_sampling_size(logger)
    #plot_acquisition_value_with_sampling_size(logger)

    #check_redundancy(logger)
    # analyze_acquisition_histogram(logger, block_set_fname='learning/data/may_cubes/blocks/set_a.pkl')
    #single_2block_tower_analysis(logger)
    #inspect_2block_towers(logger)

    #check_validation_robustness()

    #min_contact_regret_evaluation(logger)#, block_set='learning/data/block_set_10.pkl')
    #tallest_tower_regret_evaluation(logger)
    #longest_overhang_regret_evaluation(logger)#, block_set='learning/data/block_set_10.pkl')
    #tallest_tower_regret_evaluation(logger, block_set='learning/data/block_set_1000.pkl')
    #plot_regret(logger)
    # plot_constructability_over_time(logger)
    #validate(logger, 125)
    #get_stability_composition(logger, tx=245)
    #validate_by_stability_type(logger, tx=85)
    # get_percent_stable(logger)

    #save_collected_tower_images(logger)



