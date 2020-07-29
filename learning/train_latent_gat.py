
""" Massachusetts Institute of Technology

Izzy Brand, 2020
"""
from matplotlib import pyplot as plt
import numpy as np
import pickle
from random import shuffle
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from learning.gat import FCGAT

def load_dataset(name):
    """ Load all the tower data into TensorDatasets. We need a different
    dataset for each tower size, because vectorized Graph Attention Network
    can only ingest batches of graphs with equal numbers of nodes.

    Arguments:
        name {string} -- dataset name

    Returns:
        list(TensorDataset) -- datasets for each tower size
    """
    with open(f'learning/data/{name}', 'rb') as f:
        all_data = pickle.load(f)

    datasets = []
    for num_blocks in range(2,6):
        data = all_data[f'{num_blocks}block']
        # load the tower data
        towers = torch.Tensor(data['towers'])
        labels = torch.Tensor(data['labels'])

        # convert absolute xy positions to relative positions
        towers[:,1:,7:9] -= towers[:,:-1,7:9]
        # remove the three color channels at the end of each block encoding
        # and remove the COM values (these will be inferred in the latents)
        # (see block_utils.Object.vectorize for details)
        # drop indices [1,2,3] (COM) and [14,15,16] (color)
        indices_to_keep = [0,4,5,6,7,8,9,10,11,12,13]
        towers = towers[...,indices_to_keep]

        # data['block_names'] is a list of lists of strings. each sublist
        # is a tower and the strings are the names of blocks in that tower.
        # convert list(list(string)) -> Tensor(int)
        block_ids = [[int(name.strip('obj_')) for name in tower] \
            for tower in data['block_names']]
        block_ids = torch.tensor(block_ids)

        # add the new dataset to the list of datasets
        datasets.append(TensorDataset(towers, labels, block_ids))

    return datasets

def split(datasets, part_train=0.95):
    train_datasets = []
    test_datasets = []
    for d in datasets:
        # get the size of the dataset
        N = len(d)
        num_train = int(N * part_train)
        num_test = N - num_train
        d_train, d_test = torch.utils.data.random_split(d, [num_train, num_test])
        train_datasets.append(d_train)
        test_datasets.append(d_test)

    return train_datasets, test_datasets

def attach_latents_to_towers(model, towers, block_ids):
    latents = model.latents[block_ids]
    return torch.cat([towers, latents], axis=2)

def train(model, datasets):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 50
    losses = []
    num_data_points = len(datasets[0])

    for epoch_idx in range(10):
        # create a dataloader for each tower size
        iterable_dataloaders = [
            iter(DataLoader(d, batch_size=batch_size, shuffle=True))
            for d in datasets]

        for batch_idx in range(num_data_points // batch_size):
            # shuffle(iterable_dataloaders)
            # iterate through the tower sizes in the inner loop
            for iterable_dataloader in iterable_dataloaders:
                optimizer.zero_grad()

                towers, labels, bk_ids = next(iterable_dataloader)
                towers_with_latents = \
                    attach_latents_to_towers(model, towers, bk_ids)
                preds = model.iterate(towers_with_latents, k=6)
                l = F.binary_cross_entropy(preds, labels)

                l.backward()
                optimizer.step()
                accuracy = ((preds>0.5) == labels).float().mean()
                losses.append(accuracy.item())

            if batch_idx % 40 == 0:
                print(f'Epoch {epoch_idx}\tBatch {batch_idx}:\t {losses[-4:]}')

    return losses

def test(model, datasets):
    accuracies = []

    # iterate through the tower sizes
    for dataset in datasets:
        # pull out the input and output tensors for the whole dataset
        towers = dataset[:][0]
        labels = dataset[:][1]
        bk_ids = dataset[:][2]
        # run the model on everything
        towers_with_latents = \
                    attach_latents_to_towers(model, towers, bk_ids)
        preds = model.iterate(towers_with_latents, k=6)
        # calculate the and save the accuracy
        accuracy = ((preds>0.5) == labels).float().mean()
        accuracies.append(accuracy.item())

    return accuracies


if __name__ == '__main__':
    # number of blocks in the training and test set
    num_blocks = 10
    #the dimensionality of the observed block attributes
    O = 11
    # the dimensionality of the latent space
    L = 3
    # the number of hidden variables in the graph NN
    M = 20
    model = FCGAT(O+L+M, M)

    # add a latent parameter to the model for each block
    model.latents = nn.Parameter(torch.randn(num_blocks, L))

    # load the data and split into train and test
    datasets = load_dataset(f'{num_blocks}block_set_(x10000).pkl')
    train_datasets, test_datasets = split(datasets)

    losses = train(model, train_datasets)
    plt.plot(losses)
    plt.xlabel('Batch (x10)')
    plt.show()

    accuracies = test(model, test_datasets)
    plt.scatter(np.arange(2, 6), accuracies)
    plt.xlabel('Num Blocks in Tower')
    plt.ylabel('Accuracy')
    plt.show()
