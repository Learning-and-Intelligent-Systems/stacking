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
    
    with open(logger.get_figure_path('val_accuracies.pkl'), 'wb') as handle:
        pickle.dump(accs, handle)
    return accs

def plot_val_accuracy(logger):
    with open(logger.get_figure_path('val_accuracies.pkl'), 'rb') as handle:
        accs = pickle.load(handle)

    tower_keys = ['2block', '3block', '4block', '5block']

    ref_acc = {'2block': .955, '3block': .925, '4block': .912, '5block': .913}
    
    fig, axes = plt.subplots(4, figsize=(10, 20))
    for ix, ax in enumerate(axes):
        k = tower_keys[ix]
        max_x = 400 + 10*len(accs[k])
        xs = np.arange(400, max_x, 10)

        ax.plot(xs, accs[k], label=k)
        ax.plot([400, 4500], [ref_acc[k]]*2)
        ax.axvline(x=4375)
        ax.set_xlabel('Number of Towers')
        ax.set_ylabel('Val Accuracy')
        ax.legend()
    plt.savefig(logger.get_figure_path('val_accuracy.png'))

def get_dataset_statistics(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    aq_over_time = np.zeros((logger.args.max_acquisitions, 4))

    for tx in range(logger.args.max_acquisitions):
        acquired_data, _ = logger.load_acquisition_data(tx)
        
        nums = [acquired_data[k]['towers'].shape[0] for k in tower_keys]
        pos = [np.sum(acquired_data[k]['labels']) for k in tower_keys]
        print(nums, pos)

        for kx, k in enumerate(tower_keys):
            aq_over_time[tx, kx] = acquired_data[k]['towers'].shape[0]

    

    print('Totals')
    for tx in range(logger.args.max_acquisitions):
        dataset = logger.load_dataset(tx)
        print([dataset.tower_tensors[k].shape[0] for k in tower_keys])

    max_x = 400 + 10*logger.args.max_acquisitions
    xs = np.arange(400, max_x, 10)

    w = 10
    plt.figure(figsize=(20, 10))
    plt.bar(xs, aq_over_time[:, 0], width=w, label='2block')
    for kx in range(1, len(tower_keys)):
        plt.bar(xs, aq_over_time[:, kx], bottom=np.sum(aq_over_time[:, :kx], axis=1), width=w, label=tower_keys[kx])
    
    plt.xlabel('Acquisition Step')
    plt.ylabel('Number of Collected Samples')
    plt.legend()
    plt.savefig(logger.get_figure_path('acquisition_breakdown.png'))

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

    tx = 60
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

def get_acquisition_scores_over_time(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    scores = {k: [] for k in tower_keys}

    for tx in range(0, logger.args.max_acquisitions):
        if tx % 10 == 0:
            print(tx)
        _, unlabeled = logger.load_acquisition_data(tx)
        ensemble = logger.get_ensemble(tx)

        preds = get_predictions(unlabeled, ensemble)
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
        
        for tx in range(200, logger.args.max_acquisitions):
            print(tx)
            ensemble = logger.get_ensemble(tx)
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
                    it_scores[k].append(bald_scores[start:end][acquire_indices[0]])
                    start = end
            for k in tower_keys:
                overall_data[n_samples]['median'][k].append(np.median(it_scores[k]))
                overall_data[n_samples]['lower'][k].append(np.quantile(it_scores[k], 0.05))
                overall_data[n_samples]['upper'][k].append(np.quantile(it_scores[k], 0.95))

    
            with open(logger.get_figure_path('time_acquisitions.pkl'), 'wb') as handle:
                pickle.dump(overall_data, handle)

def plot_acquisition_value_with_sampling_size(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    with open(logger.get_figure_path('time_acquisitions.pkl'), 'rb') as handle:
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
            axes[kx].legend()
    plt.savefig(logger.get_figure_path('bald_vs_sample_size.png'))

    
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
            preds = get_predictions(unlabeled, ensemble)
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

    ensemble = logger.get_ensemble(logger.args.max_acquisitions - 1)
    unlabeled = sample_unlabeled_data(10000)
    preds = get_predictions(unlabeled, ensemble)
    bald_scores = bald(preds).numpy()
    
    
    preds2 = preds[:unlabeled['2block']['towers'].shape[0], :]
    bald_scores2 = bald_scores[:unlabeled['2block']['towers'].shape[0]]
    acquire_indices = np.argsort(bald_scores2)[::-1][:10]
    for ix in range(preds2.shape[0]):
        print(np.around(preds2[ix,:].numpy(), 2), np.around(bald_scores2[ix], 3))
    print('-----')
    for ix in acquire_indices:
        d = decision_distance(unlabeled['2block']['towers'][ix,:,:])
        print(np.around(preds2[ix,:].numpy(), 4), np.around(bald_scores2[ix], 3), d)

    for ix in acquire_indices:
        unlabeled['2block']['towers'][ix,1,7:8] += 0.00
    new_preds = get_predictions(unlabeled, ensemble)
    print('-----')
    for ix in acquire_indices:
        d = decision_distance(unlabeled['2block']['towers'][ix,:,:])
        print(np.around(new_preds[ix,:].numpy(), 2))
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

    preds = get_predictions(val_towers, ensemble).mean(1).numpy()

    start = 0
    for k in tower_keys:
        end = start + val_towers[k]['towers'].shape[0]
        acc = ((preds[start:end]>0.5) == val_towers[k]['labels']).mean()
        accs[k].append(acc)
        start = end
    print(accs)


def single_2block_tower_analysis(logger):
    tower_keys = ['2block', '3block', '4block', '5block']
    ensemble = logger.get_ensemble(logger.args.max_acquisitions - 1)
    unlabeled = sample_unlabeled_data(1000)
    tower = unlabeled['2block']['towers'][0:1,:,:].copy()
    displacements = np.linspace(-0.1, 0.1, 1000).reshape(1000,1,1)
    unlabeled['2block']['towers'] = np.resize(tower, (1000, 2, 17))
    unlabeled['2block']['labels'] = np.zeros((1000,))
    unlabeled['2block']['towers'][:,:,7:8] += displacements

    preds = get_predictions(unlabeled, ensemble)[:1000,...]

    for ix in range(1000):
        dim_x = unlabeled['2block']['towers'][ix, 0, 4]/2.
        com_x = unlabeled['2block']['towers'][ix, 1, 1] + unlabeled['2block']['towers'][ix, 1, 7]
        print(np.around(preds[ix,:], 3), dim_x, com_x)
    bald_scores = bald(preds).numpy()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    args = parser.parse_args()
    
    logger = ActiveExperimentLogger(args.exp_path)
    logger.args.max_acquisitions = 224
    #plot_sample_efficiency(logger)
    #analyze_sample_efficiency(logger, 340)
    #analyze_bald_scores(logger)
    #get_acquisition_scores_over_time(logger)
    #plot_acquisition_scores_over_time(logger)
    #analyze_single_dataset(logger)
    # get_dataset_statistics(logger)
    # accs = get_validation_accuracy(logger,
    #                               'learning/data/random_blocks_(x2000)_5blocks_uniform_mass.pkl')
    # plot_val_accuracy(logger)
    # analyze_collected_2block_towers(logger)
    # print(accs)

    #analyze_acquisition_value_with_sampling_size(logger)
    #plot_acquisition_value_with_sampling_size(logger)

    single_2block_tower_analysis(logger)
    #inspect_2block_towers(logger)
