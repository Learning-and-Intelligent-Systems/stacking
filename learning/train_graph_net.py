
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

from learning.domains.towers.analyze_data import is_geometrically_stable, is_com_stable, get_geometric_thresholds
from learning.domains.towers.tower_data import TowerDataset, TowerSampler
from learning.models.gat import FCGAT
from learning.models.gn import FCGN
from learning.models.mlp import MLP
from learning.models.lstm import TowerLSTM
from learning.models.gated_gn import GatedGN
from learning.models.topdown_net import TopDownNet

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



def load_dataset(name, augment=False, K=1):
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

    return TowerDataset(all_data, augment=augment, K_skip=K)


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


def train(model, dataset, test_dataset=None, epochs=100, is_ensemble=False):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 32
    losses = []
    model.training = True

    train_loader = DataLoader(dataset=dataset,
                              batch_sampler=TowerSampler(dataset=dataset,
                                                         batch_size=batch_size,
                                                         shuffle=True))

    for epoch_idx in range(epochs):
        # create a dataloader for each tower size
        accs = {2: [], 3: [], 4: [], 5: []}
        for batch_idx, (towers, labels) in enumerate(train_loader):
            model.train(True)
            optimizer.zero_grad()
            if torch.cuda.is_available():
                towers = towers.cuda()
                labels = labels.cuda()
            preds = model.forward(towers)

            if is_ensemble:
                # NOTE(izzy): do I run into any weird gradient magnitude issues if
                # i combine the losses for every model in the ensemble? Pretty
                # sure I shuold use sum, because mean would reduce the magnitude
                # of the gradient for each model
                labels = labels[:, None, ...].expand(-1, model.ensemble_size)
                scale = model.ensemble_size
            else:
                scale = 1.

            l = F.binary_cross_entropy(preds, labels) * scale
            l.backward()
            optimizer.step()

            n_blocks = towers.shape[1]
            accuracy = ((preds>0.5) == labels).float().mean()
            accs[n_blocks].append(accuracy.item())

            losses = [np.mean(accs[k][-500:]) for k in range(2, 6)]

            if batch_idx % 40 == 0:
                print(f'Epoch {epoch_idx}\tBatch {batch_idx}:\t {losses}')

        if test_dataset is not None and epoch_idx % 5 == 0:
            model.train(False)
            accuracies = test(model, test_dataset, is_ensemble=is_ensemble)
            print('Val:', accuracies)

    return losses

def test(model, dataset, fname='', is_ensemble=False):
    model.train(False) # turn off dropout before computing accuracy
    accuracies = []

    results = []
    # iterate through the tower sizes
    for k in range(2, 6):
        # pull out the input and output tensors for the whole dataset
        towers = dataset.tower_tensors['%dblock' % k]
        labels = dataset.tower_labels['%dblock' % k]
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

    #train_dataset = load_dataset('random_blocks_(x40000)_5blocks_all.pkl', augment=True, K=32)
    train_dataset = load_dataset('random_blocks_(x40000)_5blocks_uniform_mass.pkl', K=32,  augment=True)
    print('Number of Training Towers')
    print(len(train_dataset))

    #test_dataset = load_dataset('random_blocks_(x2000)_5blocks_all.pkl', augment=False)
    test_dataset = load_dataset('random_blocks_(x2000)_5blocks_uniform_mass.pkl', augment=False)
    #train_datasets = load_dataset('random_blocks_(x5000)_5blocks_pwu.pkl')
    losses = train(model, train_dataset, test_dataset, epochs=500)
    plt.plot(losses)
    plt.xlabel('Batch (x10)')
    plt.show()

    #test_datasets = load_dataset('random_blocks_(x2000)_quat.pkl')
    accuracies = test(model, test_dataset, fname='lstm_preds.pkl')
    print(accuracies)
    plt.scatter(np.arange(2,6), accuracies)
    plt.xlabel('Num Blocks in Tower')
    plt.ylabel('Accuracy')
    plt.show()
