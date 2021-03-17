import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import pickle
from random import shuffle
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from learning.domains.towers.analyze_data import is_geometrically_stable, is_com_stable, get_geometric_thresholds
from learning.domains.towers.tower_data import TowerDataset, TowerSampler
from learning.models.gat import FCGAT


class LatentEnsemble(nn.Module):
    def __init__(ensemble, latents):
        super(LatentEnsemble, self).__init__()

        self.ensemble = ensemble
        self.latents = latents

    def associate(self, samples, block_ids):
        """ given samples from the latent space for each block in the set,
        reorder them so they line up with blocks in towers

        Arguments:
            samples {torch.Tensor} -- [N_samples x N_blockset x latent_dim]
            block_ids {torch.Tensor} -- [N_batch x N_blocks]

        Returns:
            torch.Tensor -- [N_batch x N_samples x N_blocks x latent_dim]
        """
        return samples[:, block_ids, :].permute(1, 0, 2, 3)

    def concat_samples(self, samples, observed):
        """ concatentate samples from the latent space for each tower
        with the observed variables for each tower

        Arguments:
            samples {torch.Tensor} -- [N_batch x N_samples x N_blocks x latent_dim]
            observed {torch.Tensor} -- [N_batch x N_blocks x observed_dim]

        Returns:
            torch.Tensor -- [N_batch x N_samples x N_blocks x total_dim]
        """
        N_batch, N_samples, N_blocks, latent_dim = samples.shape
        observed = torch.unsqueze(observed, 1)
        observed = torch.tile(observed, (1, N_samples, 1, 1))
        return torch.cat([samples, observed], 3)

    def forward(data, ensemble_idx=None, N_samples=1):
        """ predict feasibility of the towers

            data = {
                'towers':   [N_batch x N_blocks x N_features]
                'block_id': [N_batch x N_blocks]
                'labels':   [N_batch]
            }

        Arguments:
            data {dict} -- collection of towers, block_ids, labels

        Keyword Arguments:
            ensemble_idx {int} -- if None, average of all models (default: {None})
            N_samples {number} -- how many times to sample the latents (default: {1})
        """

        samples_for_each_block_in_set = latents.sample(N_samples)
        samples_for_each_tower_in_batch = self.associate(
            samples_for_each_block_in_set, data['block_ids'])
        towers_with_latents = self.concat_samples(
            samples_for_each_tower_in_batch, data['towers'])

        N_batch, N_samples, N_blocks, total_dim = towers_with_latents.shape

        if ensemble_idx is None:
            # return vector of all predictions for the ensmble on the batch
            # [N_models x N_batch x N_samples]
            pass
        else:
            # forward pass of a single model in the ensemble
            # [N_batch x N_samples]
            pass


def update_params(latent_ensemble, batches):
    """
    1. sample ~ latents
    2. samples -(model)-> likelihood
    3. gradient descent step on model params

    Arguments:
        model {[type]} -- [description]
        latents {[type]} -- [description]
        batch {[type]} -- note that each model in the ensemble gets its own batch
    """

    likelihood_loss = 0
    for i, batch in enumerate(set_of_batches):
        likelihood_loss += latent_ensemble(batch, ensemble_idx=i)

    pass

def update_latents(latent_ensemble, batch):
    """
    [mu, sigma] -(reparam)-> [sample] -(thru ensemble)-> [likelihood]
    [mu, sigma] -> [KL]

    gradient descent step on model params [likelihood + KL]

    Choices:
        * either sample a single model ~ ensemble
        * take mean of likelihood under ensemble

    Arguments:
        ensemble {[type]} -- [description]
        latents {[type]} -- [description]
        batch {[type]} -- [description]
    """

    likelihood_loss = latent_ensemble(data) # take the mean of the ensemble
    kl_loss = latent_ensemble.latents.kl() # compute divergence of latent distribution

    pass

def train(latent_ensemble, dataset, n_epochs=10):

    optimizer = optim.Adam([ensemble.parameters(), latents.parameters()])

    for epoch_idx in range(n_epochs):
        for data in dataset:
            update_latents(ensemble, latents, data)
            update_params(ensemble, latents, data)

def test(ensemble, latents, dataset):
    update_latents(ensemble, latents, data)


if __name__ == "__main__":
    # load data
    with open("learning/data/10block_set_(x10000).pkl", 'rb') as handle:
        train_towers_dict = pickle.load(handle)
    train_dataset = TowerDataset(train_towers_dict, augment=True)
    with open("learning/data/10block_set_(x1000).pkl", 'rb') as handle:
        test_towers_dict = pickle.load(handle)
    test_dataset = TowerDataset(test_towers_dict, augment=True)

    # create the model
    # NOTE: we need to specify latent dimension.
    ensemble = None

    # create the latents
    # NOTE: we need to say how many blocks
    train_latents = None
    test_latents = None

    # train
    train(ensemble, train_latents, train_dataset)

    # test
    test(ensemble, test_latents, test_dataset)