
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

# the number of latent variables in the graph NN
M = 20

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
        # remove the three color channels at the end of each block encoding
        # (see block_utils.Object.vectorize for details)
        towers = towers[...,:14]
        # convert absolute xy positions to relative positions
        towers[:,1:,7:9] -= towers[:,:-1,7:9]
        # add the new dataset to the list of datasets
        datasets.append(TensorDataset(towers, labels))

    return datasets

def run_model_on_towers(model, towers):
    """ runs the given GAT model on the given set of vectorized towers.

    Arguments:
        model {GAT} -- the model
        towers {torch.Tensor} -- [N x K x 14] tensor of towers

    Returns:
        torch.Tensor -- [N] tensor of stability predictions
    """
    N, K, _ = towers.shape

    # create M additional channels to be used in the processing of the tower
    x = 1e-2*torch.randn(N, K, M)
    # run the network as many times as there are blocks
    for _ in range(K+2):
    # for _ in range(6):
        # append the tower information
        x = torch.cat([towers, x], axis=2)
        x = model(x)


    tower_preds = model.output(x)
    # # pull out the logit for the predicted stability of each block
    # block_preds = torch.sigmoid(x[...,-1])
    # # the tower stability involves every block being stable
    # tower_preds = block_preds.prod(axis=1)
    return tower_preds



def train(model, datasets):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 10
    losses = []
    num_data_points = len(datasets[0])

    for epoch_idx in range(5):
        # create a dataloader for each tower size
        iterable_dataloaders = [
            iter(DataLoader(d, batch_size=batch_size, shuffle=True))
            for d in datasets]

        for batch_idx in range(num_data_points // batch_size):
            # shuffle(iterable_dataloaders)
            # iterate through the tower sizes in the inner loop
            for iterable_dataloader in iterable_dataloaders:
                optimizer.zero_grad()

                towers, labels = next(iterable_dataloader)
                preds = run_model_on_towers(model, towers)
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
        # run the model on everything
        preds = run_model_on_towers(model, towers)
        # calculate the and save the accuracy
        accuracy = ((preds>0.5) == labels).float().mean()
        accuracies.append(accuracy.item())

    return accuracies


if __name__ == '__main__':
    model = FCGAT(14+M, M)

    train_datasets = load_dataset('10block_set_(x1000).pkl')
    losses = train(model, train_datasets)
    plt.plot(losses)
    plt.xlabel('Batch (x10)')
    plt.show()

    test_datasets = load_dataset('10block_set_(x1000).pkl')
    accuracies = test(model, test_datasets)
    plt.scatter(np.arange(2,6), accuracies)
    plt.xlabel('Num Blocks in Tower')
    plt.ylabel('Accuracy')
    plt.show()
