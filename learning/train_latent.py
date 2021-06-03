import argparse
from matplotlib import pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.spatial.transform import Rotation
import torch
import copy
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from learning.active.utils import ActiveExperimentLogger
from learning.domains.towers.tower_data import TowerDataset, ParallelDataLoader
from learning.models.gn import FCGN
from learning.models.ensemble import Ensemble
from learning.models.latent_ensemble import LatentEnsemble
from learning.viz_latents import viz_latents


def get_both_loss(latent_ensemble, batches, disable_latents, N, N_samples=10):
    likelihood_loss = 0
    N_models = latent_ensemble.ensemble.n_models

    for i, batch in enumerate(batches): # for each model === for each sample from the model distribution
        towers, block_ids, labels = batch
        N_batch = towers.shape[0]

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
            preds = latent_ensemble(towers[:,:,4:], block_ids.long(), ensemble_idx=i, collapse_latents=False, collapse_ensemble=False, N_samples=N_samples)
            # print(preds.shape)
        likelihood_loss += F.binary_cross_entropy(preds.squeeze(), labels[:, None].expand(N_batch, N_samples), reduction='sum')

    likelihood_loss = likelihood_loss/N_models/N_samples

    q_z = torch.distributions.normal.Normal(latent_ensemble.latent_locs, torch.exp(latent_ensemble.latent_logscales))
    p_z = torch.distributions.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
    kl_loss = torch.distributions.kl_divergence(q_z, p_z).sum()

    return (kl_loss + N*likelihood_loss)/N_batch


def get_params_loss(latent_ensemble, batches, disable_latents, N):
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
    for i, batch in enumerate(batches): # for each model === for each sample from the model distribution
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
            preds = latent_ensemble(towers[:,:,4:], block_ids.long(), collapse_latents=True, collapse_ensemble=False, ensemble_idx=i)
        likelihood_loss += F.binary_cross_entropy(preds.squeeze(), labels.squeeze(), reduction='sum')

    # we sum the likelihoods for every input in the batch, but we want the
    # expected likelihood under the ensemble which means we take the mean
    return N*likelihood_loss/towers.shape[0]/latent_ensemble.ensemble.n_models


def get_latent_loss(latent_ensemble, batch, N, N_samples=10):
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

    # old version: takes the mean of the labels before computing likelihood
    # preds = latent_ensemble(towers[:,:,4:], block_ids.long(), collapse_latents=True, collapse_ensemble=True)#, np.random.randint(0, len(latent_ensemble.ensemble.models))) # take the mean of the ensemble
    # likelihood_loss = F.binary_cross_entropy(preds.squeeze(), labels.squeeze(), reduction='sum')
    # and compute the kl divergence

    # updated version June 3: we want to take the expectation outside the likelihood
    preds = latent_ensemble(towers[:,:,4:], block_ids.long(), collapse_latents=False, collapse_ensemble=True, N_samples=N_samples)
    likelihood_losses = F.binary_cross_entropy(preds.squeeze(), labels[:, None].expand(towers.shape[0], N_samples), reduction='none')
    likelihood_loss = likelihood_losses.mean(axis=1).sum(axis=0)

    # Option 1: Calculate KL for every latent in each batch.
    q_z = torch.distributions.normal.Normal(latent_ensemble.latent_locs, torch.exp(latent_ensemble.latent_logscales))
    p_z = torch.distributions.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))

    kl_loss = torch.distributions.kl_divergence(q_z, p_z).sum()

    return (kl_loss + N*likelihood_loss)/towers.shape[0]

def train(dataloader, val_dataloader, latent_ensemble, n_epochs=30,
    freeze_latents=False,
    freeze_ensemble=False,
    disable_latents=False,
    return_logs=False,
    alternate=False):

    params_optimizer = optim.Adam(latent_ensemble.ensemble.parameters(), lr=1e-3)
    latent_optimizer = optim.Adam([latent_ensemble.latent_locs, latent_ensemble.latent_logscales], lr=1e-3)

    losses = []
    latents = []

    best_weights = None
    best_loss = 1000
    for epoch_idx in range(n_epochs):
        accs = []
        for batch_idx, set_of_batches in enumerate(dataloader):

            if alternate: # take gradient descent steps separately
                batch_loss = 0
                # update the latent distribution while holding the model parameters fixed.
                if (not freeze_latents) and (not disable_latents):
                    latent_optimizer.zero_grad()
                    latent_loss = get_latent_loss(latent_ensemble, set_of_batches[0], N=len(dataloader.loaders[0].dataset))
                    latent_loss.backward()
                    latent_optimizer.step()
                    batch_loss += latent_loss.item()

                # # update the model parameters while sampling from the latent distribution.
                if not freeze_ensemble:
                    params_optimizer.zero_grad()
                    params_loss = get_params_loss(latent_ensemble, set_of_batches, disable_latents, N=len(dataloader.loaders[0].dataset))
                    params_loss.backward()
                    params_optimizer.step()
                    batch_loss += params_loss.item()

            else: # take gradient descent steps together
                params_optimizer.zero_grad()
                latent_optimizer.zero_grad()
                both_loss = get_both_loss(latent_ensemble, set_of_batches, disable_latents, N=len(dataloader.loaders[0].dataset))
                both_loss.backward()
                if (not freeze_latents) and (not disable_latents): latent_optimizer.step()
                if not freeze_ensemble: params_optimizer.step()
                batch_loss = both_loss.item()

            losses.append(batch_loss)

        #TODO: Check for early stopping.
        if val_dataloader is not None:
            val_loss = evaluate(latent_ensemble, val_dataloader, disable_latents=disable_latents)
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(latent_ensemble.state_dict())
                print('Saved.')
            print(f'Epoch {epoch_idx}')
            # print('Train Accuracy:')
            # for k, v in compute_accuracies(latent_ensemble, dataloader, disable_latents=disable_latents).items():
            #     print(k, np.mean(v))
            print('Val Accuracy:')
            for k, v in compute_accuracies(latent_ensemble, val_dataloader, disable_latents=disable_latents).items():
                print(k, np.mean(v))
        # print(latent_ensemble.latent_locs, latent_ensemble.latent_scales)
        latents.append(np.hstack([latent_ensemble.latent_locs.cpu().detach().numpy(),
                                  torch.exp(latent_ensemble.latent_logscales).cpu().detach().numpy()]))

    if val_dataloader is not None:
        latent_ensemble.load_state_dict(best_weights)
    if return_logs:
        return latent_ensemble, losses, latents
    else:
        return latent_ensemble

    # Note (Mike): When doing active learning, add new towers to train_dataset (not train_loader).

def evaluate(latent_ensemble, data_loader, disable_latents, val_metric='f1'):
    acc = []
    losses = []

    preds = []
    labels = []
    for val_batches in data_loader:
        towers, block_ids, label = val_batches[0]
        if torch.cuda.is_available():
            towers = towers.cuda()
            block_ids = block_ids.cuda()
            label = label.cuda()
        if disable_latents:
            pred = latent_ensemble.ensemble.forward(towers).squeeze()
        else:
            pred = latent_ensemble(towers[:,:,4:], block_ids.long()).squeeze()
        if len(pred.shape) == 0: pred = pred.unsqueeze(-1)
        loss = F.binary_cross_entropy(pred, label)

        with torch.no_grad():
            preds += (pred > 0.5).cpu().float().numpy().tolist()
            labels += label.cpu().numpy().tolist()
    if val_metric == 'loss':
        score = np.mean(losses)
    else:
        score = -f1_score(labels, preds)


    return score

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
                preds = latent_ensemble(towers[:,:,4:], block_ids.long()).squeeze()
            acc = ((preds > 0.5) == labels).float().mean().item()
            accs[towers.shape[1]].append(acc)

    return accs

def test(latent_ensemble, train_loader, test_loader, disable_latents, n_epochs=50):
    latent_ensemble.reset_latents(random=False)

    print('Test Accuracy with prior latents:')
    for k, v in compute_accuracies(latent_ensemble, test_loader, disable_latents=disable_latents).items():
        print(k, '%.4f' % np.mean(v))
    # print(latent_ensemble.latent_locs, latent_ensemble.latent_scales)

    # estimate the latents for the test data, but without updating the model
    # parameters
    latent_ensemble, losses, latents = train(train_loader, None, latent_ensemble, n_epochs=n_epochs, freeze_ensemble=True, disable_latents=disable_latents, return_logs=True)
    with torch.no_grad():
        viz_latents(latent_ensemble.latent_locs.cpu().detach(), torch.exp(latent_ensemble.latent_logscales).cpu().detach())
    # np.save('learning/experiments/logs/latents/fit_during_test.npy', latents)

    print('Test Accuracy with posterior latents:')
    for k, v in compute_accuracies(latent_ensemble, test_loader, disable_latents=disable_latents).items():
        print(k, '%.4f' % np.mean(v))
    # print(latent_ensemble.latent_locs, latent_ensemble.latent_scales)

def shrink_dict(tower_dict, skip):
    for k in tower_dict.keys():
        tower_dict[k] = {
            'towers': tower_dict[k]['towers'][::skip, ...],
            'labels': tower_dict[k]['labels'][::skip, ...],
            'block_ids': tower_dict[k]['block_ids'][::skip, ...],
        }
    return tower_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-latents', action='store_true', default=False)
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')
    args = parser.parse_args()
    args.exp_name = 'pretrained_latents'
    args.use_latents = not args.disable_latents

    logger = ActiveExperimentLogger.setup_experiment_directory(args)

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
    # train_data_filename = 'learning/data/10block_set_(x1000)_train_cube1_dict.pkl'
    # test_tower_filename = 'learning/data/10block_set_(x1000)_train_cube2_dict.pkl'
    # test_block_filename = 'learning/data/10block_set_(x1000)_test_cube_dict.pkl'

    # train_data_filename = 'learning/data/10block_set_(x1000)_train_seq_dict.pkl'
    # test_tower_filename = 'learning/data/10block_set_(x1000)_train_seq2_dict.pkl'
    # test_block_filename = 'learning/data/10block_set_(x1000)_test_seq_dict.pkl'

    # train_data_filename = 'learning/data/10block_set_(x1000)_train_fixed_cube1_dict.pkl'
    # test_tower_filename = 'learning/data/10block_set_(x1000)_train_fixed_cube2_dict.pkl'
    # test_block_filename = 'learning/data/10block_set_(x1000)_test_fixed_cube_dict.pkl'

    # train_data_filename = 'learning/data/10block_set_(x1000)_cubes_train_seq1_dict.pkl'
    # test_tower_filename = 'learning/data/10block_set_(x1000)_cubes_train_seq2_dict.pkl'
    # test_block_filename = 'learning/data/10block_set_(x1000)_cubes_test_seq1_dict.pkl'

    # Datasets for blocks with dynamic poses.
    # train_block_train_tower_fname = 'learning/data/10block_set_(x1000)_blocks_a_1_dict.pkl'
    #train_block_fit_tower_fname = 'learning/data/10block_set_(x1000)_blocks_a_2_dict.pkl'
    #train_block_test_tower_fname = 'learning/data/10block_set_(x1000)_blocks_a_3_dict.pkl'
    #train_block_train_tower_fname = 'learning/data/may_blocks/towers/100block_set_(x10000)_nblocks_a_1_dict.pkl'
    # train_block_fit_tower_fname = 'learning/data/may_blocks/towers/10block_set_(x1000)_nblocks_a_2_dict.pkl'
    # train_block_test_tower_fname = 'learning/data/may_blocks/towers/10block_set_(x1000)_nblocks_a_3_dict.pkl'
    # test_block_fit_tower_fname = 'learning/data/10block_set_(x1000)_blocks_b_1_dict.pkl'
    # test_block_test_tower_fname = 'learning/data/10block_set_(x1000)_blocks_b_2_dict.pkl'

    # Datasets for cubes with fixed poses.
    # train_block_train_tower_fname = 'learning/data/10block_set_(x1000)_cubes_fixed_a_1_dict.pkl'
    # train_block_fit_tower_fname = 'learning/data/10block_set_(x1000)_cubes_fixed_a_2_dict.pkl'
    # train_block_test_tower_fname = 'learning/data/10block_set_(x1000)_cubes_fixed_a_3_dict.pkl'
    # test_block_fit_tower_fname = 'learning/data/10block_set_(x1000)_cubes_fixed_b_1_dict.pkl'
    # test_block_test_tower_fname = 'learning/data/10block_set_(x1000)_cubes_fixed_b_2_dict.pkl'

    # Datasets for cubes with dynamic poses.
    # train_block_train_tower_fname = 'learning/data/may_cubes/towers/10block_set_(x1000)_seq_a_1_dict.pkl'
    # train_block_fit_tower_fname = 'learning/data/may_cubes/towers/10block_set_(x1000)_seq_a_2_dict.pkl'
    # train_block_test_tower_fname = 'learning/data/may_cubes/towers/10block_set_(x1000)_seq_a_3_dict.pkl'
    # test_block_fit_tower_fname = 'learning/data/may_cubes/towers/10block_set_(x1000)_seq_b_1_dict.pkl'
    # test_block_test_tower_fname = 'learning/data/may_cubes/towers/10block_set_(x1000)_seq_b_2_dict.pkl'

    # Datasets for blocks with dynamic poses.
    train_block_train_tower_fname = 'learning/data/may_blocks/towers/10block_set_(x1000)_seq_a_1_dict.pkl'
    train_block_fit_tower_fname = 'learning/data/may_blocks/towers/10block_set_(x1000)_seq_a_2_dict.pkl'
    train_block_test_tower_fname = 'learning/data/may_blocks/towers/10block_set_(x1000)_seq_a_3_dict.pkl'
    test_block_fit_tower_fname = 'learning/data/may_blocks/towers/10block_set_(x1000)_seq_b_1_dict.pkl'
    test_block_test_tower_fname = 'learning/data/may_blocks/towers/10block_set_(x1000)_seq_b_2_dict.pkl'

    #train_data_filename = "learning/data/10block_set_(x4000.0)_train_10_prerotated.pkl"
    #test_tower_filename = "learning/data/10block_set_(x1000.0)_train_10_towers_prerotated.pkl"

    with open(train_block_train_tower_fname, 'rb') as handle:
        train_block_train_tower_dict = pickle.load(handle)
    with open(train_block_fit_tower_fname, 'rb') as handle:
        train_block_fit_tower_dict = pickle.load(handle)
    with open(train_block_test_tower_fname, 'rb') as handle:
        train_block_test_tower_dict = pickle.load(handle)
    with open(test_block_fit_tower_fname, 'rb') as handle:
        test_block_fit_tower_dict = pickle.load(handle)
    with open(test_block_test_tower_fname, 'rb') as handle:
        test_block_test_tower_dict = pickle.load(handle)

    train_block_fit_tower_dict = shrink_dict(train_block_fit_tower_dict, 5)
    test_block_fit_tower_dict = shrink_dict(test_block_fit_tower_dict, 5)

    # with open('learning/experiments/logs/exp-20210518-181538/datasets/active_34.pkl', 'rb') as handle:
    #     train_dataset = pickle.load(handle)

    train_block_train_tower_dataset = TowerDataset(train_block_train_tower_dict, augment=True, prerotated=False)
    train_block_fit_tower_dataset = TowerDataset(train_block_fit_tower_dict, augment=True, prerotated=False)
    train_block_test_tower_dataset = TowerDataset(train_block_test_tower_dict, augment=False, prerotated=False)
    test_block_fit_tower_dataset = TowerDataset(test_block_fit_tower_dict, augment=True, prerotated=False)
    test_block_test_tower_dataset = TowerDataset(test_block_test_tower_dict, augment=False, prerotated=False)

    logger.save_dataset(train_block_train_tower_dataset, 0)
    logger.save_val_dataset(train_block_fit_tower_dataset, 0)

    train_block_train_tower_loader = ParallelDataLoader(dataset=train_block_train_tower_dataset,
                                      batch_size=16,
                                      shuffle=True,
                                      n_dataloaders=n_models)
    train_block_fit_tower_loader = ParallelDataLoader(dataset=train_block_fit_tower_dataset,
                                           batch_size=16,
                                           shuffle=False,
                                           n_dataloaders=1)
    train_block_test_tower_loader = ParallelDataLoader(dataset=train_block_test_tower_dataset,
                                            batch_size=16,
                                            shuffle=False,
                                            n_dataloaders=1)
    test_block_fit_tower_loader = ParallelDataLoader(dataset=test_block_fit_tower_dataset,
                                            batch_size=16,
                                            shuffle=False,
                                            n_dataloaders=1)
    test_block_test_tower_loader = ParallelDataLoader(dataset=test_block_test_tower_dataset,
                                            batch_size=16,
                                            shuffle=False,
                                            n_dataloaders=1)

    ensemble = Ensemble(base_model=FCGN,
                        base_args={'n_hidden': 64, 'n_in': 10 + d_latents},
                        n_models=n_models)
    latent_ensemble = LatentEnsemble(ensemble, n_latents=n_latents, d_latents=d_latents)
    if torch.cuda.is_available():
        latent_ensemble = latent_ensemble.cuda()

    # train
    latent_ensemble.reset_latents(random=False)
    latent_ensemble, losses, latents = train(train_block_train_tower_loader, train_block_train_tower_loader, latent_ensemble, n_epochs=30, disable_latents=args.disable_latents, return_logs=True)
    logger.save_ensemble(latent_ensemble, 0)
    torch.save(latent_ensemble.state_dict(), model_path)
    np.save('learning/experiments/logs/latents/fit_during_train.npy', latents)
    print(latent_ensemble.latent_locs, torch.exp(latent_ensemble.latent_logscales))

    # test
    print('\nTesting with training blocks on training towers')
    latent_ensemble.load_state_dict(torch.load(model_path))
    test(latent_ensemble, train_block_train_tower_loader, train_block_train_tower_loader, disable_latents=args.disable_latents, n_epochs=10)

    print('\nTesting with training blocks on new towers')
    latent_ensemble.load_state_dict(torch.load(model_path))
    test(latent_ensemble, train_block_fit_tower_loader, train_block_test_tower_loader, disable_latents=args.disable_latents, n_epochs=100)

    print('\nTesting with test blocks')
    latent_ensemble.load_state_dict(torch.load(model_path))
    test(latent_ensemble, test_block_fit_tower_loader, test_block_test_tower_loader, disable_latents=args.disable_latents, n_epochs=100)




