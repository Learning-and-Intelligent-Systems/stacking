import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from learning.active.acquire import bald_diagonal_gaussian
from learning.active.utils import ActiveExperimentLogger
from learning.evaluate.active_evaluate_towers import plot_latent_uncertainty
from learning.domains.throwing.train_latent import evaluate, get_predictions
from learning.domains.throwing.throwing_data import generate_dataset, xs_to_actions
from learning.domains.throwing.task import eval_hit_target
from learning.domains.throwing.entities import ThrowingBall
from learning.domains.throwing.sanity_checking.plot_model_vs_data import generate_grid_dataset


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


def plot_latents_throughout_training(latents):
    mu, log_sigma = np.split(latents, 2, axis=-1)
    for l_idx in range(mu.shape[1]):
        for d_idx in range(mu.shape[2]):
            m = mu[:, l_idx, d_idx]
            s = np.exp(log_sigma[:, l_idx, d_idx])
            plt.plot(m)
            plt.fill_between(np.arange(m.size), m-s, m+s, alpha=0.2)

    plt.title('Mean and Variance of 5, one-dimensional latents')
    plt.ylabel('Latent value')
    plt.xlabel('Epoch')
    plt.show()


def visualize_bald_throughout_training(logger):
    objects = logger.get_objects(ThrowingBall)
    n_objects = len(objects)
    n_ang = 32
    n_w = 32
    ang_points = np.linspace(0, np.pi/2, n_ang)
    w_points = np.linspace(-10, 10, n_w)
    grid_data_tuple = generate_grid_dataset(objects, ang_points, w_points, label=False)


    for tx in range(10):
        latent_ensemble = logger.get_ensemble(tx)

        dataset = logger.load_dataset(tx)
        xs, z_ids, ys = dataset.tensors

        mu, sigma = get_predictions(latent_ensemble,
                                    grid_data_tuple,
                                    n_latent_samples=10,
                                    marginalize_latents=True,
                                    marginalize_ensemble=False,
                                    hide_dims=[3])

        scores = bald_diagonal_gaussian(sigma).numpy()
        scores = scores.reshape(n_objects, n_ang, n_w)

        fig, axes = plt.subplots(ncols=n_objects)
        for i in range(n_objects):
            # plot the BALD image
            axes[i].imshow(scores[i],
                           extent=[np.pi/8, 3*np.pi/8, -10, 10],
                           aspect='auto',
                           vmin=-3,
                           vmax=3)
            # axes[i].set_title('BALD Scores')
            # axes[i].set_xlabel('Spin')
            # axes[i].set_ylabel('Angle')


            # pull out the throwing data for this object
            xs_for_this_object = xs[z_ids == i]
            a = xs_to_actions(xs_for_this_object)
            axes[i].scatter(*a.T, c='r', s=3)
            axes[i].set_axis_off()

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, default="")
    parser.add_argument('--use-latents', action='store_true')
    parser.add_argument('--latents-log', type=str, default="")
    args = parser.parse_args()
    print(args.latents_log)

    if args.latents_log != "":
        latents = np.load(args.latents_log)
        plot_latents_throughout_training(latents)
    elif args.exp_path != "":
        logger = ActiveExperimentLogger(args.exp_path, use_latents=True)
        logger.args.max_acquisitions = 40  # lazy
        logger.args.throwing = True # lazy

        # ax = plt.gca()

        # plot_latent_uncertainty(logger, ax=ax)

        # plot_val_accuracy(logger, ax=ax)

        # objects = logger.get_objects(ThrowingBall)
        # task_score_fn = lambda latent_ensemble: eval_hit_target(latent_ensemble, objects)
        # plot_task_performance(logger, task_score_fn, ax=ax)

        visualize_bald_throughout_training(logger)
