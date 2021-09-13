import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from learning.active.utils import ActiveExperimentLogger
from learning.models.ensemble import Ensemble
from learning.models.mlp import FeedForward
from learning.models.latent_ensemble import ThrowingLatentEnsemble
from learning.domains.throwing.entities import ThrowingBall, ThrowingAction
from learning.domains.throwing.throwing_data import construct_xs, label_actions, generate_objects, ParallelDataLoader
from learning.domains.throwing.train_latent import train, get_predictions

def generate_grid_dataset(objects, ang_points, w_points, label=True):
    # produce a list of throws for a grid of initial conditions
    actions = []
    z_ids = []
    for z_id, o in enumerate(objects):
        for ang in ang_points:
            for w in w_points:
                actions.append([ang, w])
                z_ids.append(z_id)

    # create a dataset
    xs = construct_xs(objects, actions, z_ids)
    if label:
        ys = label_actions(objects, actions, z_ids)
        dataset = xs, z_ids, ys
    else:
        dataset = xs, z_ids

    return tuple(torch.Tensor(d) for d in dataset)

def generate_grid_dataset_varying_objects(ang, w):
    # produce a grid dataset with one throw and many different objects
    objects = []
    for b in np.linspace(0.1, 0.8, 32):
        for r in np.linspace(0.02, 0.06, 32):
            objects.append(ThrowingBall(bounciness=b, radius=r))

    actions = np.ones([len(objects),2])
    actions[:, 0] *= ang
    actions[:, 1] *= w

    z_ids = np.arange(len(objects))

    # create a dataset
    xs = construct_xs(objects, actions, z_ids)
    ys = label_actions(objects, actions, z_ids)

    dataset = xs, z_ids, ys
    return tuple(torch.Tensor(d) for d in dataset)


def plot_grid_for_throw(ang, w):
    print('Generating Datatset')
    dataset = generate_grid_dataset_varying_objects(ang, w)
    plt.imshow(dataset[2].numpy().reshape(32, 32), extent=[0.02, 0.06, 0.5, 1.5], aspect='auto')
    plt.colorbar()
    plt.xlabel('Radius (cm)')
    plt.ylabel('Coeff of Restitution')
    plt.title(f'Distances for throw ang: {ang}, w: {w}')
    filename = f'learning/domains/throwing/sanity_checking/figures/varying_object_parameters/ang_{ang}_w_{w}.png'
    plt.savefig(filename)
    print('saved to', filename)
    plt.clf()

def visualize_grid_data(ax, ang_points, w_points, zs, title=None):
    grid = zs.squeeze().numpy().reshape(ang_points.size, w_points.size)

    # ax.set_xlabel('W')
    # ax.set_ylabel('Angle')
    if title is not None: ax.set_title(title)
    im = ax.imshow(grid,
        extent=[w_points.min(), w_points.max(), ang_points.min(), ang_points.max()],
        aspect='auto')
    plt.colorbar(im, ax=ax)
    return im

def plot_model_vs_data():
    # get commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hide-dims', type=str, default='9')
    parser.add_argument('--save-train-dataset', type=str, default='')
    parser.add_argument('--n-train', type=int, default=500)
    parser.add_argument('--n-val', type=int, default=100)
    parser.add_argument('--n-objects', type=int, default=10)
    parser.add_argument('--n-models', type=int, default=10)
    parser.add_argument('--save-fig', type=str, default='model_vs_data.png')
    args = parser.parse_args()

    n_ang = 32
    n_w = 32
    ang_points = np.linspace(0, np.pi/2, n_ang)
    w_points = np.linspace(-10, 10, n_w)

    # inialize datasets and dataloaders
    print('Creating grid dataset')
    objects = generate_objects(args.n_objects)
    train_data_tuple = generate_grid_dataset(objects, ang_points, w_points)
    train_dataset = TensorDataset(*train_data_tuple)
    train_dataloader = ParallelDataLoader(dataset=train_dataset,
                                          batch_size=16,
                                          shuffle=True,
                                          n_dataloaders=args.n_models)

    # variables to defin e the latent ensemble
    n_latents = args.n_objects
    n_models = args.n_models
    d_observe = 12
    d_latents = 3
    d_pred = 2
    # produce a list of the dimensions of the object propoerties to make hidden
    hide_dims = [int(d) for d in args.hide_dims.split(',')] if args.hide_dims else []

    # initialize the LatentEnsemble
    ensemble = Ensemble(base_model=FeedForward,
                        base_args={
                                    'd_in': d_observe + d_latents - len(hide_dims),
                                    'd_out': d_pred,
                                    'h_dims': [64, 32]
                                  },
                        n_models=n_models)
    latent_ensemble = ThrowingLatentEnsemble(ensemble, n_latents=n_latents, d_latents=d_latents)
    if torch.cuda.is_available():
        latent_ensemble = latent_ensemble.cuda()

    # train the LatentEnsemble
    latent_ensemble.reset_latents(random=False)
    latent_ensemble, accs, latents = train(train_dataloader,
                                           None,
                                           latent_ensemble,
                                           n_epochs=100,
                                           return_logs=True,
                                           hide_dims=hide_dims)

    labels = train_data_tuple[2]
    mus, sigmas = get_predictions(latent_ensemble, train_data_tuple[:2], hide_dims=hide_dims)
    error = np.abs(mus.squeeze() - labels)

    # make the first index the object
    mus = mus.squeeze().reshape(args.n_objects, -1)
    sigmas = sigmas.squeeze().reshape(args.n_objects, -1)
    error = error.squeeze().reshape(args.n_objects, -1)
    labels = labels.squeeze().reshape(args.n_objects, -1)

    for i in range(args.n_objects):

        # plot the dataset in 2D
        fig, axes = plt.subplots(nrows=2, ncols=2)
        im = visualize_grid_data(axes[0,0], ang_points, w_points, labels[i], title='Label')
        im = visualize_grid_data(axes[0,1], ang_points, w_points, mus[i], title='Mu')
        im = visualize_grid_data(axes[1,0], ang_points, w_points, error[i], title='Error')
        im = visualize_grid_data(axes[1,1], ang_points, w_points, sigmas[i], title='Sigma')

        if args.save_fig == "":
            plt.show()
        else:
            plt.savefig(args.save_fig + f'_obj{i}.png')



if __name__ == '__main__':

    # for a small set of throws, make a higher resolution image of the
    # outcome when varying the object parameters
    # for ang in np.linspace(0, np.pi/2, 5):
    #     for w in np.linspace(-10, 10, 5):
    #         plot_grid_for_throw(ang, w)

    # train an ensemble, and then compare the ensemble to the GT over a range
    # of throw parameters
    plot_model_vs_data()


    
    
    