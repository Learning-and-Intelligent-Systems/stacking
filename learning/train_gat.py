
""" Massachusetts Institute of Technology

Izzy Brand, 2020
"""
from matplotlib import pyplot as plt
import numpy as np
from random import shuffle
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from learning.gat import FCGAT

M = 10

def load_datasets(num_towers=10000):
    """ Load all the tower data into TensorDatasets. One for each tower size.
    """
    datasets = []
    for num_blocks in range(2,6):
        stable = np.load(f'learning/data/stable_{num_blocks}block_(x{num_towers}).npy')
        unstable = np.load(f'learning/data/unstable_{num_blocks}block_(x{num_towers}).npy')
        stable_labels = np.ones(num_towers)
        unstable_labels =np.zeros(num_towers)
        towers = torch.FloatTensor(np.concatenate([stable, unstable], axis=0))
        labels = torch.FloatTensor(np.concatenate([stable_labels, unstable_labels], axis=0))
        datasets.append(TensorDataset(towers, labels))

    return datasets

def run_model_on_towers(model, towers):
    N, K, _ = towers.shape
    # remove the three color channels at the end of each block encoding
    # (see block_utils.Object.vectorize for details)
    towers = towers[...,:14]
    # introduce M additional channels to be used in the processing of the tower
    x = torch.cat([towers, torch.zeros(N, K, M)], axis=2)
    # run the network as many times as there are blocks
    for _ in range(K):
        x = model(x)
        x = torch.cat([towers, x], axis=2)

    # pull out the logit for the predicted stability of each block
    block_preds = torch.sigmoid(x[...,-1])
    # the tower stability involves every block being stable
    tower_preds = block_preds.prod(axis=1)
    return tower_preds



def train(model, datasets):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    for epoch_idx in range(50):
        # create a dataloader for each tower size
        dataloaders = [iter(DataLoader(d, batch_size=20, shuffle=True)) for d in datasets]
        shuffle(dataloaders)
        for batch_idx in range(10000 // 20):
            # iterate through the tower sizes in the inner loop
            for dataloader in dataloaders:
                optimizer.zero_grad()

                towers, labels = next(dataloader)
                preds = run_model_on_towers(model, towers)
                l = F.binary_cross_entropy(preds, labels)

                l.backward()
                optimizer.step()
                losses.append(l.item())

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch_idx}\tBatch {batch_idx}:\t {losses[-1]}')

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
        accuracies.append(accuracy)

    return accuracies



if __name__ == '__main__':
    model = FCGAT(14+M, M)

    train_datasets = load_datasets(10000)
    losses = train(model, train_datasets)
    plt.plot(losses)
    plt.show()

    test_datasets = load_datasets(1000)
    accuracies = test(model, test_datasets)
    plt.plot(accuracies)
    plt.show()