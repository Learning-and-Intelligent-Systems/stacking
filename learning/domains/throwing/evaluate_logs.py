import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from learning.active.utils import ActiveExperimentLogger
from learning.evaluate.active_evaluate_towers import plot_latent_uncertainty
from learning.domains.throwing.train_latent import evaluate
from learning.domains.throwing.throwing_data import generate_dataset
from learning.domains.throwing.task import eval_hit_target
from learning.domains.throwing.entities import ThrowingBall


def plot_task_performance(logger, task_score_fn, ax=plt.gca()):
    scores = []
    for tx in range(logger.args.max_acquisitions):
        latent_ensemble = logger.get_ensemble(tx)
        score = task_score_fn(latent_ensemble)
        print(f'Step {tx}. Score {score}')
        scores.append(score)

    ax.plot(scores)
    ax.set_xlabel('Acquisition Step')
    ax.set_ylabel('Task Loss (meters to target)')
    ax.set_title('Task Performance')
    np.save(logger.get_results_path('task_performance.npy'), np.array(scores))
    plt.savefig(logger.get_figure_path('task_performance.png'))
    ax.cla()


def plot_val_accuracy(logger, n_data=200, ax=plt.gca()):
    objects = logger.get_objects(ThrowingBall)

    print('Generating a validation dataset')
    val_dataset = TensorDataset(*generate_dataset(objects, n_data))
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=64)

    scores = []
    for tx in range(logger.args.max_acquisitions):
        latent_ensemble = logger.get_ensemble(tx)
        score = evaluate(latent_ensemble, val_dataloader)
        print(f'Step {tx}. Score {score}')
        scores.append(score)

    ax.plot(scores)
    ax.set_xlabel('Acquisition Step')
    ax.set_ylabel('Accuracy (likelihood of data)')
    ax.set_title('Validation Accuracy')
    np.save(logger.get_results_path('val_accuracy.npy'), np.array(scores))
    plt.savefig(logger.get_figure_path('val_accuracy.png'))
    ax.cla()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    parser.add_argument('--use-latents', action='store_true')
    args = parser.parse_args()
    
    logger = ActiveExperimentLogger(args.exp_path, use_latents=True)
    logger.args.max_acquisitions = 100  # lazy
    logger.args.throwing = True # lazy

    ax = plt.gca()

    plot_latent_uncertainty(logger, ax=ax)

    plot_val_accuracy(logger, ax=ax)

    objects = logger.get_objects(ThrowingBall)
    task_score_fn = lambda latent_ensemble: eval_hit_target(latent_ensemble, objects)
    plot_task_performance(logger, task_score_fn, ax=ax)
