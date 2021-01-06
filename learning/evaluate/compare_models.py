import matplotlib.pyplot as plt
import numpy as np
import pickle

from learning.active.utils import ActiveExperimentLogger


RANDOM_PATH = 'learning/experiments/logs/towers-con-init-random-blocks-10-fcgn-random-strat-20201210-173523'
#RANDOM_PATH = 'learning/experiments/logs/towers-con-init-random-blocks-10-fcgn-ent-strat-20201212-151528'
#RANDOM_PATH = 'learning/experiments/logs/towers-con-init-random-blocks-10-fcgn-100k-samples-bugfix-20201208-223619/'
#BALD_PATH = 'learning/experiments/logs/towers-con-init-random-blocks-10-fcgn-bald-strat-20201210-173551'
BALD_PATH = 'learning/experiments/logs/towers-con-init-random-blocks-10-fcgn-sample-bug-fixed-20201208-220629'

def plot_regret(random_logger, bald_logger, fname):
    plt.clf()
    fig, axes = plt.subplots(4, sharex=False, figsize=(10,20))
    for name, logger in zip(['Random', 'BALD'], [random_logger, bald_logger]):
        with open(logger.get_figure_path('%s.pkl' % fname), 'rb') as handle:
            regrets = pickle.load(handle)

        tower_keys = ['2block', '3block', '4block', '5block']
        upper975 = {k: [] for k in tower_keys}
        upper75 = {k: [] for k in tower_keys}
        median = {k: [] for k in tower_keys}
        lower25 = {k: [] for k in tower_keys}
        lower025 = {k: [] for k in tower_keys}
        for k in tower_keys:
            rs = regrets[k]
            print(rs)
            for tx in range(len(rs)):
                median[k].append(np.median(rs[tx]))
                lower025[k].append(np.quantile(rs[tx], 0.05))
                lower25[k].append(np.quantile(rs[tx], 0.25))
                upper75[k].append(np.quantile(rs[tx], 0.75))
                upper975[k].append(np.quantile(rs[tx], 0.95))
        
        for kx, k in enumerate(tower_keys):
            #xs = np.arange(400, 400+100*len(median[k]), 100)
            xs = np.arange(40, 40+100*len(median[k]), 100)
            #xs = np.arange(40, 40+10*len(median[k]), 10)
            axes[kx].plot(xs, median[k], label=name)
            axes[kx].fill_between(xs, lower25[k], upper75[k], alpha=0.2)
            #axes[kx].fill_between(xs, lower025[k], upper975[k], alpha=0.2)
            axes[kx].set_ylim(0.0, 1.1)
            axes[kx].set_ylabel('Regret', fontsize=20)
            axes[kx].set_xlabel('Number of Towers', fontsize=20)
            #axes[kx].set_title(k)
            axes[kx].legend(prop={'size': 20})
            axes[kx].tick_params(axis='both', which='major', labelsize=16)
    plt.savefig('learning/evaluate/plots/%s.png' % fname)

def plot_val_accuracy(random_logger, bald_logger):
    plt.clf()
    fig, axes = plt.subplots(4, figsize=(10, 20))
    for name, logger in zip(['Random', 'BALD'], [random_logger, bald_logger]):
        with open(logger.get_figure_path('val_accuracies.pkl'), 'rb') as handle:
            accs = pickle.load(handle)

        tower_keys = ['2block', '3block', '4block', '5block']

        #ref_acc = {'2block': .955, '3block': .925, '4block': .912, '5block': .913}
        
        
        for ix, ax in enumerate(axes):
            k = tower_keys[ix]
            max_x = 40 + 10*len(accs[k])
            xs = np.arange(40, max_x, 10)

            ax.plot(xs, accs[k], label=name)
            #ax.plot([400, 4500], [ref_acc[k]]*2)
            #ax.axvline(x=4375)
            ax.set_xlabel('Number of Towers', fontsize=20)
            ax.set_ylabel('Accuracy', fontsize=20)
            ax.legend(prop={'size': 20})
            ax.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig('learning/evaluate/plots/accuracy_comparison.png')

if __name__ == '__main__':
    random_logger = ActiveExperimentLogger(RANDOM_PATH)
    bald_logger = ActiveExperimentLogger(BALD_PATH)

    plot_regret(random_logger, bald_logger, fname='height_regret_blocks')
    plot_regret(random_logger, bald_logger, fname='contact_regret')
    plot_regret(random_logger, bald_logger, fname='longest_overhang')
    plot_val_accuracy(random_logger, bald_logger)


