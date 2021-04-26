import argparse
from matplotlib import pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.spatial.transform import Rotation
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from learning.domains.towers.tower_data import TowerDataset, ParallelDataLoader
from learning.models.gn import FCGN, FCGNFC
from learning.models.gated_gn import GatedGN
from learning.models.lstm import TowerLSTM
from learning.models.ensemble import Ensemble


class LatentEnsemble(nn.Module):
    def __init__(self, ensemble, n_latents, d_latents):
        """
        Arguments:
            n_latents {int}: Number of blocks.
            d_latents {int}: Dimension of each latent.
        """
        super(LatentEnsemble, self).__init__()

        self.ensemble = ensemble

        self.latent_locs = nn.Parameter(torch.zeros(n_latents, d_latents))
        self.latent_scales = nn.Parameter(torch.ones(n_latents, d_latents))
        self.latents = torch.distributions.normal.Normal(self.latent_locs, self.latent_scales)

    def reset_latents(self):
        with torch.no_grad():
            self.latent_locs.data[:] = 0.
            self.latent_scales.data[:] = 1.

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
        observed = observed.unsqueeze(1).expand(-1, N_samples, -1, -1)
        return torch.cat([samples, observed], 3)

    def prerotate_latent_samples(self, towers, samples):
        """ concatentate samples from the latent space for each tower
        with the observed variables for each tower

        Arguments:
            samples {torch.Tensor} -- [N_batch x N_samples x N_blocks x latent_dim]
            towers {torch.Tensor}   [N_batch x N_blocks x N_features]

        Returns:
            torch.Tensor -- [N_batch x N_samples x N_blocks x latent_dim]
        """
        N_batch, N_samples, N_blocks, latent_dim = samples.shape

        # pull out the quaternions for each block, and flatten the batch+block dims
        quats = towers[..., -4:].view(-1, 4)
        # create rotation matrices from the quaternions
        r = Rotation.from_quat(quats.cpu()).as_matrix()
        r = torch.Tensor(r)
        if torch.cuda.is_available(): r = r.cuda()
        # unflatten the batch+block dims and expand the sample dimension
        # now it should be [N_batch x N_samples x N_blocks x 3 x 3]
        r = r.view(N_batch, N_blocks, 3, 3).unsqueeze(1).expand(-1, N_samples, -1, -1, -1)
        # apply the rotation to the last three dimensions of the samples
        samples[...,-3:] = torch.einsum('asoij, asoj -> asoi', r, samples[...,-3:])
        # and return the result
        return samples


    def forward(self, towers, block_ids, ensemble_idx=None, N_samples=1):
        """ predict feasibility of the towers

        Arguments:
            towers {torch.Tensor}   [N_batch x N_blocks x N_features]
            block_ids {torch.Tensor}   [N_batch x N_blocks]

        Keyword Arguments:
            ensemble_idx {int} -- if None, average of all models (default: {None})
            N_samples {number} -- how many times to sample the latents (default: {1})

        Returns:
            torch.Tensor -- [N_batch]
        """
        # draw samples from the latent distribution [N_samples x N_blockset x latent_dim]
        # samples_for_each_block_in_set = self.latents.rsample(sample_shape=[N_samples])
        samples_for_each_block_in_set = self.latent_locs.unsqueeze(0).expand(N_samples, -1, -1)
        # assocate those latent samples with the blocks in the towers
        samples_for_each_tower_in_batch = self.associate(
            samples_for_each_block_in_set, block_ids)
        #samples_for_each_tower_in_batch = self.prerotate_latent_samples(towers, samples_for_each_tower_in_batch)
        towers_with_latents = self.concat_samples(
            samples_for_each_tower_in_batch, towers)

        # reshape the resulting tensor so the batch dimension holds
        # N_batch times N_samples
        N_batch, N_samples, N_blocks, total_dim = towers_with_latents.shape
        towers_with_latents = towers_with_latents.view(-1, N_blocks, total_dim)

        # forward pass of the model(s)
        if ensemble_idx is None:
            # mean prediction for the ensemble
            labels = self.ensemble.forward(towers_with_latents)
            labels = labels.mean(axis=1)
        else:
            # prediction of a single model in the ensemble
            labels = self.ensemble.models[ensemble_idx].forward(towers_with_latents)

        # reshape the result so we can compute the mean of the samples
        labels = labels.view(N_batch, N_samples)
        return labels.mean(axis=1)


def get_params_loss(latent_ensemble, batches, disable_latents):
    """
    1. sample ~ latents
    2. samples -(model)-> likelihood
    3. gradient descent step on model params

    Arguments:
        model {[type]} -- [description]
        latents {[type]} -- [description]
        batches {[type]} -- note that each model in the ensemble gets its own batch
    """

    likelihood_loss = 0
    for i, batch in enumerate(batches):
        towers, block_ids, labels = batch
        if torch.cuda.is_available():
            towers = towers.cuda()
            block_ids = block_ids.cuda()
            labels = labels.cuda()
        # TODO(izzy): I'm dropping the first four elements from the vectorized
        # towers, mass and COM xyz. I'm not sure if this is the best place to
        # do that because it it is still in the datast. It should probably be a
        # flag in the TowerDataset?
        if disable_latents:
            preds = latent_ensemble.ensemble.models[i].forward(towers).squeeze()
        else:
            preds = latent_ensemble(towers[:,:,4:], block_ids.long(), ensemble_idx=i)
        likelihood_loss += F.binary_cross_entropy(preds, labels)#, reduction='sum')

    # we sum the likelihoods for every input in the batch, but we want the
    # expected likelihood under the ensemble which means we take the mean
    return likelihood_loss#/len(batches)


def get_latent_loss(latent_ensemble, batch, beta=1):
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
    towers, block_ids, labels = batch
    if torch.cuda.is_available():
        towers = towers.cuda()
        block_ids = block_ids.cuda()
        labels = labels.cuda()
    # NOTE(izzy): we update the params of the latent distribution using the
    # reparametrization technique through a sample from that distribution. we may
    # wish to draw multiple samples from the latent distribution to reduce the
    # variance of the updates
    # TODO(izzy): I'm dropping the first four elements from the vectorized
    # towers, mass and COM xyz. I'm not sure if this is the best place to
    # do that because it it is still in the datast. It should probably be a
    # flag in the TowerDataset?
    preds = latent_ensemble(towers[:,:,4:], block_ids.long())#, np.random.randint(0, len(latent_ensemble.ensemble.models))) # take the mean of the ensemble
    likelihood_loss = F.binary_cross_entropy(preds, labels, reduction='sum')
    # and compute the kl divergence
    q_z = latent_ensemble.latents
    p_z = torch.distributions.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
    kl_loss = torch.distributions.kl_divergence(q_z, p_z).sum()
    return likelihood_loss + beta * kl_loss

def train(latent_ensemble, train_loader, n_epochs=30, freeze_latents=False, freeze_ensemble=False, print_accuracy=True, disable_latents=False):

    params_optimizer = optim.Adam(latent_ensemble.ensemble.parameters(), lr=1e-3)
    # TODO: Check if learning rate should be different for the latents.
    latent_optimizer = optim.Adam([latent_ensemble.latent_locs, latent_ensemble.latent_scales], lr=1e-3)

    # NOTE(izzy): we should be computing the KL divergence + likelihood of entire dataset.
    # so for each batch we need to divide by the number of batches
    # beta = 1./len(train_loader)
    beta = 0.0

    losses = []
    latents = []
    for epoch_idx in range(n_epochs):
        if print_accuracy: print(f'Epoch {epoch_idx}')
        accs = []
        for batch_idx, set_of_batches in enumerate(train_loader):
            batch_loss = 0
            # update the latent distribution while holding the model parameters fixed.
            if (not freeze_latents) and (not disable_latents):
                latent_optimizer.zero_grad()
                latent_loss = get_latent_loss(latent_ensemble, set_of_batches[0], beta=beta)
                latent_loss.backward()
                latent_optimizer.step()
                batch_loss += latent_loss.item()

            # update the model parameters while sampling from the latent distribution.
            if not freeze_ensemble:
                params_optimizer.zero_grad()
                params_loss = get_params_loss(latent_ensemble, set_of_batches, disable_latents)
                params_loss.backward()
                params_optimizer.step()
                batch_loss += params_loss.item()

            losses.append(batch_loss)

        if print_accuracy:
            print('Train Accuracy:')
            for k, v in compute_accuracies(latent_ensemble, train_loader, disable_latents=disable_latents).items():
                print(k, np.mean(v))
        # print(latent_ensemble.latent_locs, latent_ensemble.latent_scales)
        latents.append(np.hstack([latent_ensemble.latent_locs.cpu().detach().numpy(),
                                  latent_ensemble.latent_scales.cpu().detach().numpy()]))

    return latents

    # Note (Mike): When doing active learning, add new towers to train_dataset (not train_loader).

def compute_accuracies(latent_ensemble, data_loader, disable_latents):
    with torch.no_grad():
        accs = {2: [], 3:[], 4:[], 5:[]}
        for val_batches in data_loader:
            towers, block_ids, labels = val_batches[0]
            if torch.cuda.is_available():
                towers = towers.cuda()
                block_ids = block_ids.cuda()
                labels = labels.cuda()
            if disable_latents:
                preds = latent_ensemble.ensemble.forward(towers).squeeze()
            else: 
                preds = latent_ensemble(towers[:,:,4:], block_ids.long())
            acc = ((preds > 0.5) == labels).float().mean().item()
            accs[towers.shape[1]].append(acc)

    return accs

def test(latent_ensemble, test_loader, disable_latents):
    latent_ensemble.reset_latents()

    print('Test Accuracy with prior latents:')
    for k, v in compute_accuracies(latent_ensemble, test_loader, disable_latents=disable_latents).items():
        print(k, np.mean(v))
    #print(latent_ensemble.latent_locs, latent_ensemble.latent_scales)

    # estimate the latents for the test data, but without updating the model
    # parameters
    latents = train(latent_ensemble, test_loader, freeze_ensemble=True, print_accuracy=False, disable_latents=disable_latents)
    # np.save('learning/experiments/logs/latents/fit_during_test.npy', latents)

    print('Test Accuracy with posterior latents:')
    for k, v in compute_accuracies(latent_ensemble, test_loader, disable_latents=disable_latents).items():
        print(k, np.mean(v))
    print(latent_ensemble.latent_locs, latent_ensemble.latent_scales)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-latents', action='store_true', default=False)
    parser.add_argument('--use-onehot', action='store_true', default=False)
    args = parser.parse_args()
    # print(args)

    # sample_unlabeled to generate dataset. 50/50 class split. using 10 blocks
    # sample_sequential

    model_path = 'learning/models/pretrained/latent_ensemble_1.pt'


    # NOTE(izzy): d_latents corresponds to the number of unique objects in the
    # dataset. at some point we should write a function that figures that out
    # from the loaded data-dict
    n_models = 1
    d_latents = 4
    if args.disable_latents:
        d_latents = 4
    n_latents = 10

    # NOTE(izzy): data generated by
    # python -m learning.domains.towers.generate_tower_training_data --block-set-size=10 --suffix=test_joint --max-blocks=5

    # load data
    train_data_filename = "learning/data/10block_set_(x1000)_train_seq_dict.pkl"
    #train_data_filename = "learning/data/10block_set_(x1000)_cubes_train_seq1_dict.pkl"
    with open(train_data_filename, 'rb') as handle:
        train_dict = pickle.load(handle)
    test_tower_filename = "learning/data/10block_set_(x1000)_train_seq2_dict.pkl"
    #test_tower_filename = "learning/data/10block_set_(x1000)_cubes_train_seq2_dict.pkl"
    with open(test_tower_filename, 'rb') as handle:
        test_tower_dict = pickle.load(handle)
    test_block_filename = "learning/data/10block_set_(x1000)_test_seq_dict.pkl"
    with open(test_block_filename, 'rb') as handle:
        test_block_dict = pickle.load(handle)
    
    train_dataset = TowerDataset(train_dict, augment=True, prerotated=False, use_onehot=args.use_onehot)
    test_tower_dataset = TowerDataset(test_tower_dict, augment=False, prerotated=False, use_onehot=args.use_onehot, onehot_lookup=train_dataset.onehot_lookup)
    test_block_dataset = TowerDataset(test_block_dict, augment=False, prerotated=False, use_onehot=args.use_onehot, onehot_lookup=train_dataset.onehot_lookup)

    train_loader = ParallelDataLoader(dataset=train_dataset,
                                      batch_size=16,
                                      shuffle=True,
                                      n_dataloaders=n_models)
    test_tower_loader = ParallelDataLoader(dataset=test_tower_dataset,
                                           batch_size=16,
                                           shuffle=False,
                                           n_dataloaders=1)
    test_blocks_loader = ParallelDataLoader(dataset=test_block_dataset,
                                            batch_size=16,
                                            shuffle=False,
                                            n_dataloaders=1)

    # for bx, batch in enumerate(train_loader):
    #     print(bx, len(batch), len(train_dataset))


    # create the model
    # NOTE: we need to specify latent dimension.
    d_feat = 10
    if args.use_onehot:
        d_feat = 30

    ensemble = Ensemble(base_model=FCGN,
                        base_args={'n_hidden': 64, 'n_in': d_feat + d_latents},
                        n_models=n_models)
    latent_ensemble = LatentEnsemble(ensemble, n_latents=n_latents, d_latents=d_latents)
    if torch.cuda.is_available():
        latent_ensemble = latent_ensemble.cuda()

    # train
    latents = train(latent_ensemble, train_loader, print_accuracy=True, disable_latents=args.disable_latents)
    torch.save(latent_ensemble.state_dict(), model_path)
    np.save('learning/experiments/logs/latents/fit_during_train.npy', latents)
    print(latent_ensemble.latent_locs, latent_ensemble.latent_scales)

    # test
    print('\nTesting with training blocks on training towers')
    latent_ensemble.load_state_dict(torch.load(model_path))
    test(latent_ensemble, train_loader, disable_latents=args.disable_latents)

    print('\nTesting with training blocks on new towers')
    latent_ensemble.load_state_dict(torch.load(model_path))
    test(latent_ensemble, test_tower_loader, disable_latents=args.disable_latents)

    print('\nTesting with test blocks ')
    latent_ensemble.load_state_dict(torch.load(model_path))
    test(latent_ensemble, test_blocks_loader, disable_latents=args.disable_latents)

    
    
