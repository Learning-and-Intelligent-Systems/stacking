"""
Massachusetts Institute of Technology

Izzy Brand, 2020
"""

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from learning.dropout_gn import DropoutFCGN
from learning.train_graph_net import load_dataset, train, test


def H(x, eps=1e-6):
    """ Compute the element-wise entropy of x

    Arguments:
        x {torch.Tensor} -- array of probabilities in (0,1)

    Keyword Arguments:
        eps {float} -- prevent failure on x == 0

    Returns:
        torch.Tensor -- H(x)
    """
    return -(x+eps)*torch.log(x+eps)

def score(model, x, k=100):
    # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]

    with torch.no_grad():
        # take k monte-carlo samples of forward pass w/ dropout
        p = torch.stack([model(x) for i in range(k)], dim=1)
        # computing the mutual information requires a label distribution. the
        # model predicts probility of stable p, so the distribution is p, 1-p
        Y = torch.stack([p, 1-p], axis=2)
        H1 = H(Y.mean(axis=1)).sum(axis=1)
        H2 = H(Y).sum(axis=(1,2))/k

        return H1 - H2

def active(model, train_datasets, pool_datasets, test_datasets):
    batch_size = 128

    # score each of the datapoints in the pool
    # create a tensor to keep track of the scores
    num_tower_sizes = len(pool_datasets)
    num_towers_per_size = len(pool_datasets[0])
    scores = torch.zeros(num_tower_sizes, num_towers_per_size)
    for tower_size_idx, pool_dataset in enumerate(pool_datasets):
        pool_loader = DataLoader(pool_dataset,
            batch_size=batch_size, shuffle=False)
        for batch_idx, (towers, _) in enumerate(pool_loader):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + towers.shape[0]
            scores[tower_size_idx, start_idx:end_idx] = score(model, towers)

            print(f'Tower Size {tower_size_idx}\t batch {batch_idx}/{num_towers_per_size // batch_size}')

    return scores


if __name__ == '__main__':
    model = DropoutFCGN(14, 64)

    num_train = 5000 # number of pretrain examples (for each tower size)
    num_pool = 1000  # the size of the pool (for active learning)

    if torch.cuda.is_available():
        model = model.cuda()


    # load the datasets
    # datasets are actually a List(TensorDataset), one for each tower size.
    # differing tower sizes requires tensors of differing dimensions
    train_and_pool_datasets = load_dataset('random_blocks_(x40000)_5blocks_all.pkl')
    test_datasets = load_dataset('random_blocks_(x2000)_5blocks_all.pkl')

    # pick the indices at which to split into to train and pool
    # NOTE(izzy): this assumes that every sub-dataset (for each tower size)
    # is the same length
    subset_indices = np.random.choice(len(train_and_pool_datasets[0]),
        size=num_train+num_pool, replace=False)
    train_indices = subset_indices[:num_train]
    pool_indices = subset_indices[-num_pool:]

    # and split each sub-dataset
    train_datasets = [torch.utils.data.Subset(d, train_indices) for d in train_and_pool_datasets]
    pool_datasets = [torch.utils.data.Subset(d, train_indices) for d in train_and_pool_datasets]

    # pretrain the model
    losses = train(model, train_datasets, test_datasets, epochs=5)
    plt.plot(losses)
    plt.xlabel('Batch (x10)')
    plt.show()

    # do active learning until the pool is empty
    scores = active(model, train_datasets, pool_datasets, test_datasets)
    import sys; sys.exit(0)

    # and do a final test of the model
    accuracies = test(model, test_datasets, fname='lstm_preds.pkl')
    print(accuracies)
    plt.scatter(np.arange(2,6), accuracies)
    plt.xlabel('Num Blocks in Tower')
    plt.ylabel('Accuracy')
    plt.show()
