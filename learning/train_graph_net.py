
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
from learning.lstm import TowerLSTM
from learning.gated_gn import GatedGN
from learning.topdown_net import TopDownNet

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

def load_dataset(name, K=1):
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
    for num_blocks in range(2, 6):
        data = all_data[f'{num_blocks}block']
        # load the tower data
        towers = torch.Tensor(data['towers'])
        labels = torch.Tensor(data['labels'])

        #towers, labels = get_subsets(data)
        # remove the three color channels at the end of each block encoding
        # (see block_utils.Object.vectorize for details)
        towers = towers[...,:14]
        #towers = towers[...,[0, 1, 2, 4, 5, 7, 8]]
        # convert absolute xy positions to relative positions
        #towers[:,1:,7:9] -= towers[:,:-1,7:9]
        #towers[:,:,1:3] += towers[:,:,7:9]
        towers[:,:,1:4] /= 0.01 #towers[:,:,4:7]
        towers[:,:,7:9] /= 0.01 #towers[:,:,4:6]
        towers[:,:,4:7] = (towers[:,:,4:7] - 0.1) / 0.01
        towers[:,:,0] = (towers[:,:,0] - 0.55)
        # add the new dataset to the list of datasets
        datasets.append(TensorDataset(towers[::K,:], labels[::K]))

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


def train(model, datasets, test_datasets, epochs=100, is_ensemble=False):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 32
    losses = []
    num_data_points = len(datasets[0])
    model.training = True

    for epoch_idx in range(epochs):
        # create a dataloader for each tower size
        iterable_dataloaders = [
            iter(DataLoader(d, batch_size=batch_size, shuffle=True))
            for d in datasets]
        accs = [[],[],[],[]]
        for batch_idx in range(num_data_points // batch_size):
            # shuffle(iterable_dataloaders)
            # iterate through the tower sizes in the inner loop
            for dx, iterable_dataloader in enumerate(iterable_dataloaders):
                model.train(True)
                optimizer.zero_grad()
                towers, labels = next(iterable_dataloader)
                if torch.cuda.is_available():
                    towers = towers.cuda()
                    labels = labels.cuda()
                preds = model.forward(towers, k=towers.shape[1]-1)

                if is_ensemble:
                    # NOTE(izzy): do I run into any weird gradient magnitude issues if
                    # i combine the losses for every model in the ensemble? Pretty
                    # sure I shuold use sum, because mean would reduce the magnitude
                    # of the gradient for each model
                    labels = labels[:, None, ...].expand(-1, model.ensemble_size)
                    scale = model.ensemble_size
                else:
                    scale = 1

                l = F.binary_cross_entropy(preds, labels) * scale
                l.backward()
                optimizer.step()

                model.train(False) # turn off dropout before computing accuracy
                accuracy = ((preds>0.5) == labels).float().mean()
                accs[dx].append(accuracy.item())
                losses.append(np.mean(accs[dx][-500:]))
                

            if batch_idx % 40 == 0:
                print(f'Epoch {epoch_idx}\tBatch {batch_idx}:\t {losses[-4:]}')

        if epoch_idx % 5 == 0:
            model.train(False) # turn off dropout before computing accuracy
            accuracies = test(model, test_datasets, fname='lstm_preds.pkl', is_ensemble=is_ensemble)
            print('Val:', accuracies)

        #print_split_accuracies(datasets[0], model)
    return losses

def test(model, datasets, fname='', is_ensemble=False):
    model.training = False
    accuracies = []
    
    results = []
    # iterate through the tower sizes
    for dataset in datasets:
        # pull out the input and output tensors for the whole dataset
        towers = dataset[:][0]
        labels = dataset[:][1]
        if torch.cuda.is_available():
            towers = towers.cuda()
            labels = labels.cuda()
        # run the model on everything
        with torch.no_grad():
            preds = model.forward(towers, k=towers.shape[1]-1)

        if is_ensemble:
            labels = labels[:, None, ...].expand(-1, model.ensemble_size)

        # calculate the and save the accuracy
        accuracy = ((preds>0.5) == labels).float().mean()
        accuracies.append(accuracy.item())
        results.append((towers.cpu(), labels.cpu(), preds.cpu()))
    if len(fname) > 0:
        with open(fname, 'wb') as handle:
            pickle.dump(results, handle)
    return accuracies


if __name__ == '__main__':
    # the number of hidden variables in the graph NN
    M = 64
    #model = FCGAT(14+M, M)
    #model = MLP(3, 256)
    model = FCGN(14, 64)
    #model = TowerLSTM(14, 32)
    #model = GatedGN(14, 64, n_layers=2)
    #model = TopDownNet(14, 32)
    if torch.cuda.is_available():
        model = model.cuda()

    #train_datasets = load_dataset('random_blocks_(x40000)_5blocks_all.pkl')
    train_datasets = load_dataset('random_blocks_(x40000)_5blocks_uniform_mass_aug_4.pkl', K=1)
    print('Number of Training Towers') 
    for d in train_datasets:
        print(len(d))
    #test_datasets = load_dataset('random_blocks_(x2000)_5blocks_all.pkl')
    test_datasets = load_dataset('random_blocks_(x2000)_5blocks_uniform_mass.pkl', K=1)
    #train_datasets = load_dataset('random_blocks_(x5000)_5blocks_pwu.pkl')
    losses = train(model, train_datasets, test_datasets, epochs=500)
    plt.plot(losses)
    plt.xlabel('Batch (x10)')
    plt.show()

    #test_datasets = load_dataset('random_blocks_(x2000)_quat.pkl')
    accuracies = test(model, test_datasets, fname='lstm_preds.pkl')
    print(accuracies)
    plt.scatter(np.arange(2,6), accuracies)
    plt.xlabel('Num Blocks in Tower')
    plt.ylabel('Accuracy')
    plt.show()
