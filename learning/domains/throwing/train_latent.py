import argparse
import copy
from matplotlib import pyplot as plt
import numpy as np
import pickle
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from learning.models.ensemble import Ensemble
from learning.models.mlp import FeedForward
from learning.models.latent_ensemble import ThrowingLatentEnsemble
from learning.domains.throwing.throwing_data import generate_objects, generate_dataset, ParallelDataLoader, preprocess_batch, postprocess_pred, parse_hide_dims


def get_predictions(latent_ensemble,
                    unlabeled_data,
                    n_latent_samples=10,
                    marginalize_latents=True,
                    marginalize_ensemble=True,
                    hide_dims=[],
                    use_normalization=True,
                    return_normalized=False):

    dataset = TensorDataset(*unlabeled_data)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=64)

    mus = []
    sigmas = []

    with torch.no_grad():
        normed = preprocess_batch(unlabeled_data, hide_dims,
            normalize_x=use_normalization,
            normalize_y=True) # doesn't matter, because there is no y
        x = normed[0]
        z_id = normed[1]
        pred = latent_ensemble(x, z_id.long(),
                               collapse_latents=marginalize_latents,
                               collapse_ensemble=marginalize_ensemble,
                               N_samples=n_latent_samples).squeeze()

        mu, sigma = postprocess_pred(pred, unnormalize=(use_normalization and not return_normalized))

    return mu, sigma


def get_both_loss(latent_ensemble,
                  batches,
                  N,
                  n_latent_samples=2,
                  hide_dims=[],
                  use_normalization=True):
    """ compute the loglikelohood of both the latents and the ensemble

    Arguments:
        latent_ensemble {ThrowingLatentEnsemble} -- [description]
        batches {list(torch.Tensor)} -- [description]
        N {int} -- total number of training examples

    Keyword Arguments:
        n_latent_samples {number} -- number of samples from z (default: {10})
    """

    likelihood_loss = 0
    N_models = latent_ensemble.ensemble.n_models
    loss_func = nn.GaussianNLLLoss(reduction='sum', full=True)

    for i, batch in enumerate(batches):
        x, z_id, y = preprocess_batch(batch, hide_dims,
            normalize_x=use_normalization,
            normalize_y=use_normalization)
        N_batch = x.shape[0]

        # run a forward pass of the network
        pred = latent_ensemble(x, z_id.long(), ensemble_idx=i, collapse_latents=False, collapse_ensemble=False, N_samples=n_latent_samples).squeeze(dim=1)

        # and compute the likeliehood of y (no need to un-normalize, because we want to compute the loss in the normalized space)
        mu, sigma = postprocess_pred(pred, unnormalize=False)
        likelihood_loss += loss_func(mu, y[:, None, None].expand(N_batch, n_latent_samples, 1), sigma**2)


    likelihood_loss = likelihood_loss/N_models/n_latent_samples

    q_z = torch.distributions.normal.Normal(latent_ensemble.latent_locs, torch.exp(latent_ensemble.latent_logscales))
    p_z = torch.distributions.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
    kl_loss = torch.distributions.kl_divergence(q_z, p_z).sum()

    return (kl_loss + N*likelihood_loss)/N_batch


def evaluate(latent_ensemble,
            dataloader,
            hide_dims=[],
            use_normalization=True,
            likelihood=True,
            rmse=True,
            l1=True,
            var=True):
    """ computes the data likelihood

    Arguments:
        latent_ensemble {[type]} -- [description]
        dataloader {[type]} -- [description]

    Keyword Arguments:
        normalized {bool} -- [description] (default: {True})

    Returns:
        [type] -- [description]
    """

    nll_func = nn.GaussianNLLLoss(reduction='mean', full=True) # log-liklihood loss
    mse_func = nn.MSELoss(reduction='mean')
    l1_func = nn.L1Loss(reduction='mean')

    # mu, sigma = get_predictions(latent_ensemble, dataloader.dataset.tensors,
    #     hide_dims=hide_dims,
    #     use_normalization=use_normalization)

    mu, sigma = get_predictions(latent_ensemble, dataloader.dataset.tensors,
        hide_dims=hide_dims,
        marginalize_ensemble=False,
        use_normalization=use_normalization)

    # expand y to match the shape of mu and sigma
    m_y = dataloader.dataset.tensors[2]
    y = torch.tile(m_y[:, None], [1, latent_ensemble.ensemble.n_models])

    stats = np.array([
        -nll_func(mu.squeeze(), y, sigma.squeeze()**2),
        torch.sqrt(mse_func(y, mu.squeeze())),
        l1_func(y, mu.squeeze()),
        (sigma**2).mean()
    ])

    return stats[[likelihood, rmse, l1, var]]


def train(dataloader, val_dataloader, latent_ensemble,
    n_epochs=30,
    freeze_latents=False,
    freeze_ensemble=False,
    return_logs=False,
    hide_dims=[],
    use_normalization=True):

    params_optimizer = optim.Adam(latent_ensemble.ensemble.parameters(), lr=1e-3)
    latent_optimizer = optim.Adam([latent_ensemble.latent_locs, latent_ensemble.latent_logscales], lr=1e-3)

    scores = []
    latents = []
    best_weights = None
    best_L = -np.inf

    for epoch_idx in range(n_epochs):
        print(f'Epoch {epoch_idx}')

        for batch_idx, set_of_batches in enumerate(dataloader):
            params_optimizer.zero_grad()
            latent_optimizer.zero_grad()
            both_loss = get_both_loss(latent_ensemble, set_of_batches,
                N=len(dataloader.loaders[0].dataset),
                n_latent_samples=10,
                hide_dims=hide_dims,
                use_normalization=use_normalization)

            both_loss.backward()
            if not freeze_latents: latent_optimizer.step()
            if not freeze_ensemble: params_optimizer.step()
            batch_loss = both_loss.item()


        if val_dataloader is not None:
            val_score = evaluate(latent_ensemble, val_dataloader,
                hide_dims=hide_dims,
                use_normalization=use_normalization)

            scores.append(val_score)

            # early stopping based on log likelihood
            L = val_score[0].item() # pull out the log-likelihood score
            if L > best_L:
                best_L = L
                best_weights = copy.deepcopy(latent_ensemble.state_dict())
                print('New best validation score.', L)

        latents.append(np.hstack([latent_ensemble.latent_locs.cpu().detach().numpy(),
                                  torch.exp(latent_ensemble.latent_logscales).cpu().detach().numpy()]))

    if val_dataloader is not None:
        latent_ensemble.load_state_dict(best_weights)
    if return_logs:
        return latent_ensemble, scores, np.array(latents)
    else:
        return latent_ensemble


def generate_or_load_datasets(args):
    # get datasets
    if args.train_dataset == '' and args.val_dataset == '':
        train_objects = generate_objects(args.n_objects)
        print('Generating Training Data')
        train_data_tuple = generate_dataset(train_objects, args.n_train)
        print('Generating Validation Data')
        val_data_tuple = generate_dataset(train_objects, args.n_val)

        if args.save_train_dataset != '':
            print('Saving training data to', args.save_train_dataset)
            with open(args.save_train_dataset, 'wb') as handle:
                pickle.dump(train_data_tuple, handle)
        if args.save_val_dataset != '':
            print('Saving vaidation data to', args.save_val_dataset)
            with open(args.save_val_dataset, 'wb') as handle:
                pickle.dump(val_data_tuple, handle)

    elif args.train_dataset != '' and args.val_dataset != '':
        print('Loading Training Data')
        with open(args.train_dataset, 'rb') as handle:
            train_data_tuple = pickle.load(handle)
        print('Loading Validation Data')
        with open(args.val_dataset, 'rb') as handle:
            val_data_tuple = pickle.load(handle)
        print('Warning: Training and Validation data should use same block set.')

    else:
        print('Both train and val datasets must be specified to load from file.')

    # inialize datasets and dataloaders
    train_dataset = TensorDataset(*train_data_tuple)
    val_dataset = TensorDataset(*val_data_tuple)
    train_dataloader = ParallelDataLoader(dataset=train_dataset,
                                          batch_size=16,
                                          shuffle=True,
                                          n_dataloaders=args.n_models)
    val_dataloader = ParallelDataLoader(dataset=val_dataset,
                                        batch_size=16,
                                        shuffle=False,
                                        n_dataloaders=1)

    return train_dataloader, val_dataloader


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hide-dims', type=str, default='9')
    parser.add_argument('--train-dataset', type=str, default='')
    parser.add_argument('--val-dataset', type=str, default='')
    parser.add_argument('--save-train-dataset', type=str, default='')
    parser.add_argument('--save-val-dataset', type=str, default='')
    parser.add_argument('--save-accs', type=str, default='')
    parser.add_argument('--save-latents', type=str, default='')
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--n-train', type=int, default=500)
    parser.add_argument('--n-val', type=int, default=100)
    parser.add_argument('--n-objects', type=int, default=10)
    parser.add_argument('--n-models', type=int, default=10)
    parser.add_argument('--d-latent', type=int, default=2)
    parser.add_argument('--use-normalization', action='store_true')

    return parser


def main(args):
    # get the datasets
    train_dataloader, val_dataloader = generate_or_load_datasets(args)

    # variables to define the latent ensemble
    n_latents = args.n_objects
    n_models = args.n_models
    d_latents = args.d_latent
    d_observe = 12
    d_pred = 2
    # produce a list of the dimensions of the object propoerties to make hidden
    hide_dims = parse_hide_dims(args.hide_dims)

    # initialize the LatentEnsemble
    ensemble = Ensemble(base_model=FeedForward,
                        base_args={
                                    'd_in': d_observe + d_latents - len(hide_dims),
                                    'd_out': d_pred,
                                    'h_dims': [16, 64]
                                  },
                        n_models=n_models)
    latent_ensemble = ThrowingLatentEnsemble(ensemble, n_latents=n_latents, d_latents=d_latents)
    if torch.cuda.is_available():
        latent_ensemble = latent_ensemble.cuda()

    # train the LatentEnsemble
    latent_ensemble.reset_latents(random=False)
    latent_ensemble, scores, latents = train(train_dataloader,
                                           val_dataloader,
                                           latent_ensemble,
                                           n_epochs=args.n_epochs,
                                           return_logs=True,
                                           hide_dims=hide_dims,
                                           use_normalization=args.use_normalization)


    if args.save_accs != "":
        print('Saving accuracy data to', args.save_accs)
        np.save(args.save_accs, np.array(accs))
    else:
        scores = np.array(scores)
        for data, label in zip(scores.T, ["NLL", "RMSE", "L1", "Var"]):
            plt.plot(data, label=label)

        plt.plot((latents[:,:,-d_latents:]).mean(axis=1), label="Latent Var")
        plt.legend()
        plt.title('Layer sizes: ' + str(ensemble.base_args['h_dims']))
        plt.show()

    if args.save_latents != "":
        print('Saving latents data to', args.save_latents)
        np.save(args.save_latents, np.array(latents))


if __name__ == '__main__':
    # get commandline arguments
    parser = get_parser()
    args = parser.parse_args()
    main(args)
