import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from learning.active.acquire import bald_diagonal_gaussian
from learning.active.utils import ActiveExperimentLogger
from learning.evaluate.active_evaluate_towers import plot_latent_uncertainty
from learning.domains.throwing.train_latent import evaluate, get_predictions
from learning.domains.throwing.throwing_data import generate_dataset, xs_to_actions, parse_hide_dims
from learning.domains.throwing.task import eval_hit_target
from learning.domains.throwing.entities import ThrowingBall
from learning.domains.throwing.sanity_checking.plot_model_vs_data import generate_grid_dataset
from learning.models.latent_ensemble import ThrowingLatentEnsemble, PFThrowingLatentEnsemble


def plot_task_performance(logger, task_score_fn, ax=plt.gca()):
    print("Plot task performance throughout training")
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
    print("Plot validation accuracy throughout training")
    objects = logger.get_objects(ThrowingBall)

    print('Generating a validation dataset')
    val_dataset = TensorDataset(*generate_dataset(objects, n_data))
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=64)

    scores = []
    for tx in range(logger.args.max_acquisitions):
        latent_ensemble = logger.get_ensemble(tx)
        score = evaluate(latent_ensemble, val_dataloader,
            hide_dims=parse_hide_dims(logger.args.hide_dims),
            use_normalization=logger.args.use_normalization,
            return_rmse=True)
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
    print("Plot latents throughout training")
    mu, log_scales = np.split(latents, 2, axis=-1)
    for l_idx in range(mu.shape[1]):
        for d_idx in range(mu.shape[2]):
            m = mu[:, l_idx, d_idx]
            s = np.sqrt(np.exp(log_scales[:, l_idx, d_idx]))
            plt.plot(m)
            plt.fill_between(np.arange(m.size), m-s, m+s, alpha=0.2)

    plt.title('Mean and Variance of 5, one-dimensional latents')
    plt.ylabel('Latent value')
    plt.xlabel('Epoch')
    plt.show()


def plot_pf_uncertainty(logger, ax=plt.gca()):
    print("Plot PF Uncertainty")
    scales = []

    # go through each acqisition step
    for tx in range(0, logger.args.max_acquisitions):
        print('Eval timestep, ', tx)

        # load the dataset and ensemble from that timestep
        ensemble = logger.get_ensemble(tx)
        uncertainty = np.array([np.cov(obj_particles, rowvar=False) for obj_particles in ensemble.latent_locs.detach().numpy()]).mean()
        scales.append(uncertainty)

    plt.plot(np.array(scales))

    ax.set_xlabel('Acquisition Step')
    ax.set_ylabel('Mean Latent Scale')
    ax.set_title('Variance along each latent dimension')
    plt.savefig(logger.get_figure_path('latent_scale.png'))
    ax.cla()


def visualize_bald_throughout_training(logger):
    objects = logger.get_objects(ThrowingBall)
    n_objects = len(objects)
    n_ang = 32
    n_w = 32
    ang_points = np.linspace(0, np.pi/2, n_ang)
    w_points = np.linspace(-10, 10, n_w)
    grid_data_tuple = generate_grid_dataset(objects, ang_points, w_points, label=False)


    for tx in range(logger.args.max_acquisitions):
        latent_ensemble = logger.get_ensemble(tx)

        dataset = logger.load_dataset(tx)
        xs, z_ids, ys = dataset.tensors

        mu, sigma = get_predictions(latent_ensemble,
                                    grid_data_tuple,
                                    n_latent_samples=10,
                                    marginalize_latents=True,
                                    marginalize_ensemble=False,
                                    hide_dims=[3])

        scores = bald_diagonal_gaussian(mu, sigma).numpy()
        scores = scores.reshape(n_objects, n_ang, n_w)
        print(scores.min(), scores.max())

        fig, axes = plt.subplots(ncols=n_objects)
        for i in range(n_objects):
            # plot the BALD image
            axes[i].imshow(scores[i],
                           extent=[np.pi/8, 3*np.pi/8, -10, 10],
                           aspect='auto',
                           vmin=0,
                           vmax=2)
            # axes[i].set_title('BALD Scores')
            # axes[i].set_xlabel('Spin')
            # axes[i].set_ylabel('Angle')


            # pull out the throwing data for this object
            xs_for_this_object = xs[z_ids == i]
            a = xs_to_actions(xs_for_this_object)
            axes[i].scatter(*a.T, c='r', s=3)
            axes[i].set_axis_off()

        plt.show()

def visualize_acquired_and_bald(logger, show_labels=False):
    objects = logger.get_objects(ThrowingBall)
    n_objects = len(objects)
    print('N Objects: ', n_objects)

    n_ang, n_w = 15, 15
    ang_points = np.linspace(np.pi/8, 3*np.pi/8, n_ang)
    w_points = np.linspace(-10, 10, n_w)
    
    # Returns single-dimensional list. 
    grid_data_tuple = generate_grid_dataset(objects, ang_points, w_points, label=show_labels)

    for tx in range(50, logger.args.max_acquisitions):
        fig, axes = plt.subplots(ncols=n_objects, nrows=4)
        latent_ensemble = logger.get_ensemble(tx)
        # Load the dataset.
        dataset = logger.load_dataset(tx)
        xs, z_ids, ys = dataset.tensors
        # Load the acquired data.
        acquired = logger.load_acquisition_data(tx)
        ac_xs, ac_z_ids, ac_ys = acquired[0]
        unlab_xs, unlab_z_ids = acquired[1]

        mu, sigma = get_predictions(latent_ensemble,
                                    (grid_data_tuple[0], grid_data_tuple[1]),
                                    n_latent_samples=10,
                                    marginalize_latents=True,
                                    marginalize_ensemble=False,
                                    hide_dims=[9],
                                    use_normalization=logger.args.use_normalization)
        print(sigma.min(), sigma.max())
        # Scores are in the same order as grid_data_tuple.
        scores, m_ent, ent = bald_diagonal_gaussian(mu, sigma, return_components=True, use_mc=False)
        scores, m_ent, ent = scores.numpy(), m_ent.numpy(), ent.numpy()
        print('Scores:', scores.min(), scores.max())

        for i in range(n_objects):

            img = np.zeros((n_ang, n_w))
            img_m_ent = np.zeros((n_ang, n_w))
            img_ent = np.zeros((n_ang, n_w))
            for im_x in range(0, n_ang):
                for im_y in range(0, n_w):
                    obj_start_ix = i*n_ang*n_w
                    
                    img[im_x, im_y] = scores[obj_start_ix + n_ang*im_x + im_y]
                    img_m_ent[im_x, im_y] = m_ent[obj_start_ix + n_ang*im_x + im_y]
                    img_ent[im_x, im_y] = ent[obj_start_ix + n_ang*im_x + im_y]
            
            if show_labels:
                true_img = np.zeros((n_ang, n_w))
                for im_x in range(0, n_ang):
                    for im_y in range(0, n_w):
                        obj_start_ix = i*n_ang*n_w
                        
                        true_img[im_x, im_y] = grid_data_tuple[2][obj_start_ix + n_ang*im_x + im_y]
                axes[1][i].imshow(true_img.T,
                            extent=[np.pi/8, 3*np.pi/8, -10, 10],
                            aspect='auto',
                            vmin=grid_data_tuple[2].min(),
                            vmax=grid_data_tuple[2].max(),
                            origin='lower')

            axes[0][i].imshow(img.T,
                            extent=[np.pi/8, 3*np.pi/8, -10, 10],
                            aspect='auto',
                            vmin=scores.min(),
                            vmax=scores.max(),
                            origin='lower')
            axes[2][i].imshow(img_m_ent.T,
                            extent=[np.pi/8, 3*np.pi/8, -10, 10],
                            aspect='auto',
                            vmin=m_ent.min(),
                            vmax=m_ent.max(),
                            origin='lower')
            axes[3][i].imshow(img_ent.T,
                            extent=[np.pi/8, 3*np.pi/8, -10, 10],
                            aspect='auto',
                            vmin=m_ent.min(),
                            vmax=m_ent.max(),
                            origin='lower')
            
            # pull out the throwing data for this object
            xs_for_this_object = xs[z_ids == i]
            acs_for_this_object = ac_xs[ac_z_ids == i]
            unlab_for_this_object = unlab_xs[unlab_z_ids == i]
            a = xs_to_actions(xs_for_this_object)
            ac_as = xs_to_actions(acs_for_this_object)
            unlab_as = xs_to_actions(unlab_for_this_object)
            #axes[i].scatter(*a.T, c='r', s=3)
            axes[0][i].scatter(*unlab_as.T, c='r', s=1)
            axes[0][i].scatter(*ac_as.T, c='g', s=10)

            axes[1][i].scatter(*a.T, c='r', s=3)
            print(ac_as.shape, a.shape)
            #axes[i].set_axis_off()

        plt.show()

def visualize_dataset(logger, tx):
    latent_ensemble = logger.get_ensemble(tx)

    objects = logger.get_objects(ThrowingBall)[:5]
    n_objects = len(objects)
    print('N Objects: ', n_objects)
    dataset = logger.load_dataset(tx)
    xs, z_ids, ys = dataset.tensors

    n_ang = 5
    n_w = 5
    ang_points = np.linspace(0, np.pi/2, n_ang)
    w_points = np.linspace(-20, 20, n_w)
    
    grid_data_tuple = generate_grid_dataset(objects, ang_points, w_points, label=True)
    fig, axes = plt.subplots(ncols=n_objects, nrows=2)
    for i in range(n_objects):
        labels = grid_data_tuple[2].reshape(n_objects, n_ang, n_w)
        print('Labels:', labels.min(), labels.max())

        mu, sigma = get_predictions(latent_ensemble,
                                    (grid_data_tuple[0], grid_data_tuple[1]),
                                    n_latent_samples=10,
                                    marginalize_latents=True,
                                    marginalize_ensemble=False,
                                    hide_dims=[9])

        scores = bald_diagonal_gaussian(mu, sigma).numpy()
        scores = scores.reshape(n_objects, n_ang, n_w)
        scores = sigma.mean(dim=1).reshape(n_objects, n_ang, n_w).numpy()
        print('Scores:', scores.min(), scores.max())

        axes[0][i].imshow(labels[i],
                        extent=[np.pi/8, 3*np.pi/8, -10, 10],
                        aspect='auto',
                        vmin=0,
                        vmax=2)
        axes[1][i].imshow(scores[i],
                        extent=[np.pi/8, 3*np.pi/8, -10, 10],
                        aspect='auto',
                        vmin=1,#0,
                        vmax=14)#0.7)
        # pull out the throwing data for this object
        xs_for_this_object = xs[z_ids == i]
        a = xs_to_actions(xs_for_this_object)
        axes[0][i].scatter(*a.T, c='r', s=3)
        #axes[i].set_axis_off()

    plt.show()

def plot_with_variance(x, ys, ax, c=None, label=None, alpha=0.3):
    x = np.array(x)
    ys = np.array(ys)
    # flip the ys as needed

    print(x.shape, ys.shape)
    if ys.shape[1] != x.shape[0]:
        ys = ys.T
        assert ys.shape[1] == x.shape[0], 'x and ys don\'t have matching dimensions'

    mu = np.mean(ys, axis=0)
    sigma = np.sqrt(np.var(ys, axis=0))
    ax.fill_between(x, mu-sigma, mu+sigma, color=c, alpha=alpha)
    ax.plot(x, mu, c=c, alpha=alpha, label=label)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, default="")
    parser.add_argument('--use-latents', action='store_true')
    parser.add_argument('--latents-log', type=str, default="")
    args = parser.parse_args()
    print(args.latents_log)


    if args.latents_log != "":
        #######################################################################
        # plotting non-standard log format
        #######################################################################
        latents = np.load(args.latents_log)
        plot_latents_throughout_training(latents)

    elif args.exp_path != "":
        #######################################################################
        # plotting for single logs
        #######################################################################
        logger = ActiveExperimentLogger(args.exp_path, use_latents=True)
        logger.args.max_acquisitions = 80  # lazy
        logger.args.throwing = True # lazy

        # visualize_acquired_and_bald(logger, show_labels=False)
        # sys.exit()
        ax = plt.gca()
        if isinstance(logger.get_ensemble(1), PFThrowingLatentEnsemble):
            plot_pf_uncertainty(logger, ax=ax)
        else:
            plot_latent_uncertainty(logger, ax=ax)

        plot_val_accuracy(logger, ax=ax, n_data=1000)
        sys.exit()

        objects = logger.get_objects(ThrowingBall)
        # function that predicts the outcome of each throw
        data_pred_fn = lambda latent_ensemble, xs: get_predictions(latent_ensemble, xs,
            hide_dims=parse_hide_dims(logger.args.hide_dims),
            use_normalization=logger.args.use_normalization)
        # score that prediction function for each latent ensemble
        task_score_fn = lambda latent_ensemble: eval_hit_target(latent_ensemble, objects, data_pred_fn)
        plot_task_performance(logger, task_score_fn, ax=ax)

        # visualize_bald_throughout_training(logger)
    else:
        #######################################################################
        # plotting for multipe logs
        #######################################################################

        runs = [
            {
                "prefix": 'active',
                "label": 'BALD',
                "data": [],
                "color": 'b'
            },
            {
                "prefix": 'random',
                "label": 'Random',
                "data": [],
                "color": 'r'
            },

        ]
        exp_path = 'learning/experiments/logs/new_throwing'
        ax = plt.gca()
        for r in runs:
            for fname in os.listdir(exp_path):
                if fname.startswith(r["prefix"]):
                    path_to_log = exp_path + '/' + fname
                    path_to_task_performance_file = path_to_log + '/results/task_performance.npy'
                    print(f'Loading from {fname}')
                    if not os.path.isfile(path_to_task_performance_file):
                        print(f'Failed to find task_performance.npy for {fname}. Processing Log.')
                        logger = ActiveExperimentLogger(path_to_log, use_latents=True)
                        logger.args.max_acquisitions = 50  # lazy
                        logger.args.throwing = True # lazy
                        plot_latent_uncertainty(logger, ax=ax)
                        plot_val_accuracy(logger, ax=ax)
                        objects = logger.get_objects(ThrowingBall)
                        task_score_fn = lambda latent_ensemble: eval_hit_target(latent_ensemble, objects, use_normalization=logger.args.use_normalization)
                        plot_task_performance(logger, task_score_fn, ax=ax)

                    r["data"].append(np.load(path_to_task_performance_file))


        for r in runs:
            plot_with_variance(np.arange(50), r["data"], ax, label=r["label"], c=r["color"])

        plt.legend()
        plt.xlabel('Acquisition Step')
        plt.ylabel('Task Error (m)')
        plt.title('Task Loss Throughout Active Fitting')
        plt.show()



