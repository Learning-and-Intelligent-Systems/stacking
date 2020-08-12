
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

from learning.analyze_data import is_geometrically_stable, is_com_stable, get_geometric_thresholds
from learning.gat import FCGAT
from learning.gn import FCGN
from learning.mlp import MLP


def get_subsets(data):
    towers, labels = [], []
    for ix in range(data['towers'].shape[0]):
        tower = data['towers'][ix,:,:]
        label = data['labels'][ix]
        g_stable = is_geometrically_stable(tower[0,:], tower[1,:])
        c_stable = is_com_stable(tower[0,:], tower[1,:])
        if label != c_stable:
            continue
        if (not g_stable and label) or (g_stable and not label):
            towers.append(tower)
            labels.append(label)
    return torch.Tensor(towers), torch.Tensor(labels)

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

        # towers = torch.cat([towers[0:1250, :], towers[2500:3750,:],
        #                     towers[6250:7500, :], towers[8750:, :]], dim=0)
        # labels = torch.cat([labels[0:1250], labels[2500:3750],
        #                     labels[6250:7500], labels[8750:]], dim=0)
        #towers, labels = get_subsets(data)
        # remove the three color channels at the end of each block encoding
        # (see block_utils.Object.vectorize for details)
        towers = towers[...,:14]
        # convert absolute xy positions to relative positions
        #towers[:,1:,7:9] -= towers[:,:-1,7:9]

        # add the new dataset to the list of datasets
        datasets.append(TensorDataset(towers, labels))

    return datasets

def print_split_accuracies(dataset, model):
    """
    Subdivide the accuracies into multiple groups.
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    correct_dict = {}
    count_dict = {}
    for g in [0, 1]:
        correct_dict[g] = {}
        count_dict[g] = {}
        for c in [0, 1]:
            correct_dict[g][c] = 0
            count_dict[g][c] = 0

    total = 0
    thresholds = []
    for tower, label in dataloader:
        g_stable = is_geometrically_stable(tower[0,0,:], tower[0,1,:])
        c_stable = int(label)
        #pred = model.iterate(tower, k=1)
        pred = model.forward(tower, k=tower.shape[1]-1)
        correct = ((pred > 0.5) == c_stable).float().mean()
        if correct < 0.5:
            thresholds += get_geometric_thresholds(tower[0,0,:], tower[0,1,:])

        count_dict[g_stable][c_stable] += 1
        correct_dict[g_stable][c_stable] += correct

        total += 1
        if total > 2000: break
    print(count_dict, correct_dict)
    print('Thresholds:', np.mean(thresholds))
    for g in [0, 1]:
        for c in [0, 1]:
            if count_dict[g][c] == 0:
                acc = -1
            else:
                acc = correct_dict[g][c]/count_dict[g][c]
            print('Geom %d, CoM %d: %f' % (g, c, acc))


def train(model, datasets):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 8
    losses = []
    num_data_points = len(datasets[0])

    for epoch_idx in range(1000):
        # create a dataloader for each tower size
        iterable_dataloaders = [
            iter(DataLoader(d, batch_size=batch_size, shuffle=True))
            for d in datasets]
        accs = [[],[],[],[]]
        for batch_idx in range(num_data_points // batch_size):
            # shuffle(iterable_dataloaders)
            # iterate through the tower sizes in the inner loop
            optimizer.zero_grad()
            for dx, iterable_dataloader in enumerate(iterable_dataloaders):
                

                towers, labels = next(iterable_dataloader)
                #preds = model.iterate(towers, k=1)
                preds = model.forward(towers, k=1)#towers.shape[1]-1)
                l = F.binary_cross_entropy(preds, labels)
                l.backward()
                optimizer.step()
                accuracy = ((preds>0.5) == labels).float().mean()
                accs[dx].append(accuracy)
                losses.append(np.mean(accs[dx][-500:]))
            
                #print(preds, labels)
                

            if batch_idx % 40 == 0:
                print(f'Epoch {epoch_idx}\tBatch {batch_idx}:\t {losses[-4:]}')

        #print_split_accuracies(datasets[0], model)
    return losses

def test(model, datasets):
    accuracies = []

    # iterate through the tower sizes
    for dataset in datasets:
        # pull out the input and output tensors for the whole dataset
        towers = dataset[:][0]
        labels = dataset[:][1]
        # run the model on everything
        preds = model.forward(towers, k=towers.shape[1]-1)
        # calculate the and save the accuracy
        accuracy = ((preds>0.5) == labels).float().mean()
        accuracies.append(accuracy.item())

    return accuracies


if __name__ == '__main__':
    # the number of hidden variables in the graph NN
    M = 128
    #model = FCGAT(14+M, M)
    #model = MLP(2, 128)
    model = FCGN(14, 128)

    train_datasets = load_dataset('random_blocks_(x10000)_5blocks_all.pkl')
    losses = train(model, train_datasets)
    plt.plot(losses)
    plt.xlabel('Batch (x10)')
    plt.show()

    test_datasets = load_dataset('random_blocks_(x2000)_quat.pkl')
    accuracies = test(model, test_datasets)
    plt.scatter(np.arange(2,6), accuracies)
    plt.xlabel('Num Blocks in Tower')
    plt.ylabel('Accuracy')
    plt.show()
