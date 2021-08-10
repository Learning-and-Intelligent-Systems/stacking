import pickle
import matplotlib.pyplot as plt
import numpy as np
from learning.active.utils import ActiveExperimentLogger

def plot_compare_training_phase_validation(init_dataset_size=40, n_acquire=10):
    MODELS = {
        'marginal': 'learning/experiments/logs/latents-train-marginal-bugfix-20210608-215642',
        'joint': 'learning/experiments/logs/latents-train-joint-bugfix-20210608-215630'
    }
    sizes = ['2block', '3block', '4block', '5block']
    plt.clf()
    fig, axes = plt.subplots(4, figsize=(10, 15))
    for name, fname in MODELS.items():
        logger = ActiveExperimentLogger(fname, use_latents=True)
        with open(logger.get_figure_path('val_accuracies.pkl'), 'rb') as handle:
            val_accs = pickle.load(handle)
        
        for sx, s in enumerate(sizes):
            max_x = init_dataset_size + n_acquire*len(val_accs[s])
            xs = np.arange(init_dataset_size, max_x, n_acquire)

            axes[sx].plot(xs, val_accs[s], label=name)
            axes[sx].set_xlabel('Number of Towers')
            axes[sx].set_ylabel('Val Accuracy')
            axes[sx].set_title(s)
            axes[sx].legend()
        fig.tight_layout(pad=5)
    plt.show()


def plot_compare_fitting_phase_validation(init_dataset_size=0, n_acquire=2):
    # MODELS = {
    #     'marg-marg': 'learning/experiments/logs/train-marg-fit-marg-20210609-220839',
    #     'joint-joint': 'learning/experiments/logs/train-joint-fit-joint-20210609-221233',
    #     'marg-joint': 'learning/experiments/logs/train-marg-fit-joint-20210609-220952',
    #     'joint-marg': 'learning/experiments/logs/train-joint-fit-marg-20210609-221235'
    # }
    MODELS = {
        'marg-marg': 'learning/experiments/logs/train-marg-fit-marg-early-20210610-133538',
        'joint-joint': 'learning/experiments/logs/train-joint-fit-joint-early-20210610-133628',
        'marg-joint': 'learning/experiments/logs/train-marg-fit-joint-early-20210610-133602',
        'joint-marg': 'learning/experiments/logs/train-joint-fit-marg-early-20210610-133615'
    }
    sizes = ['2block', '3block', '4block', '5block']
    plt.clf()
    fig, axes = plt.subplots(4, figsize=(10, 15))
    for name, fname in MODELS.items():
        logger = ActiveExperimentLogger(fname, use_latents=True)
        with open(logger.get_figure_path('val_accuracies.pkl'), 'rb') as handle:
            val_accs = pickle.load(handle)
        
        for sx, s in enumerate(sizes):
            max_x = init_dataset_size + n_acquire*len(val_accs[s])
            xs = np.arange(init_dataset_size, max_x, n_acquire)

            axes[sx].plot(xs, val_accs[s], label=name)
            axes[sx].set_xlabel('Number of Towers')
            axes[sx].set_ylabel('Val Accuracy')
            axes[sx].set_title(s)
            axes[sx].legend()
        fig.tight_layout(pad=5)
    plt.show()

if __name__ == '__main__':
    plot_compare_training_phase_validation()
    plot_compare_fitting_phase_validation()