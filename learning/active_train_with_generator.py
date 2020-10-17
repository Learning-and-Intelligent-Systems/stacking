"""
Massachusetts Institute of Technology

Izzy Brand, 2020
"""
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from block_utils import Object
from learning.generate_tower_training_data import build_tower, vectorize
from learning.dropout_gn import DropoutFCGN
from learning.train_graph_net import load_dataset, preprocess, train, test


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


def bald(Y):
    """ compute the bald score for the given distribution

    Arguments:
        Y {torch.Tensor} -- [batch_size x num_samples x num_classes]

    Returns:
        torch.Tensor -- bald scores for each element in the batch [batch_size]
    """
    # 1. average over the sample dimensions to get the mean class distribution
    # 2. compute the entropy of the class distribution
    H1 = H(Y.mean(axis=1)).sum(axis=1)
    # 1. compute the entropy of the sample AND class distribution
    # 2. and sum over the class dimension
    # 3. and average over the sample dimension
    H2 = H(Y).sum(axis=(1,2))/Y.shape[1]
    # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]
    return H1 - H2

def bernoulli_bald(p):
    return bald(torch.stack([p, 1-p], axis=2))

def mc_dropout_score(model, x, k=100):
    # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]

    with torch.no_grad():
        # take k monte-carlo samples of forward pass w/ dropout
        p = torch.stack([model(x) for i in range(k)], dim=1)
        # computing the mutual information requires a label distribution. the
        # model predicts probility of stable p, so the distribution is p, 1-p
        return bernoulli_bald(p)

def active(model, train_datasets, test_datasets, score_func=mc_dropout_score):
    batch_size = 32
    num_to_acquire_per_batch = 4
    num_to_acquire_in_total = 40
    for _ in range(num_to_acquire_per_batch // num_to_acquire_in_total1)
        model.train(True)

        for i in range(4):
            # generate new towers
            num_blocks = train_datasets[i][0][0].shape[0] # NOTE(izzy): I don't love this, lol
            print(f'Generating {batch_size} towers of height {num_blocks}')
            stability_attributes = (torch.rand(batch_size, 3) > 0.5)
            towers = [build_tower(num_blocks, *stablity) for stablity in stability_attributes]
            towers = torch.stack([torch.tensor(vectorize(t)) for t in towers if t is not None])
            towers = preprocess(towers)

            # score them
            if torch.cuda.is_available(): towers = towers.cuda()
            scores = mc_dropout_score(model, towers)

            # and add the best towers to the dataset
            best_idxs = np.argsort(scores)[:min(num_to_acquire_per_batch, len(scores))]
            best_towers = towers[best_idxs]
            best_labels = stability_attributes[best_idxs, 0]
            train_datasets[i] = TensorDataset(
                torch.cat([train_datasets[i][:][0], best_towers]),
                torch.cat([train_datasets[i][:][1], best_labels]))

        # model.reset()
        print('Dataset sizes:', [len(d) for d in train_datasets])
        train(model, train_datasets, test_datasets, epochs=1)


def evaluate_active_learning_performance(model, score_func, train_datasets, test_datasets):

    # pretrain the model
    losses = train(model, train_datasets, test_datasets, epochs=1)
    # plt.plot(losses)
    # plt.xlabel('Batch (x10)')
    # plt.show()

    # do active learning until the pool is empty
    scores = active(model, train_datasets, test_datasets, score_func=score_func)
    # import sys; sys.exit(0)

    # and do a final test of the model
    accuracies = test(model, test_datasets, fname='lstm_preds.pkl')
    print(accuracies)
    plt.scatter(np.arange(2,6), accuracies)
    plt.xlabel('Num Blocks in Tower')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    model = DropoutFCGN(14, 64)
    
    num_train = 1000

    if torch.cuda.is_available():
        model = model.cuda()

    # load the datasets
    # datasets are actually a List(TensorDataset), one for each tower size.
    # differing tower sizes requires tensors of differing dimensions
    train_datasets = load_dataset('random_blocks_(x40000)_5blocks_all.pkl')
    test_datasets = load_dataset('random_blocks_(x2000)_5blocks_all.pkl')
    subset_idxs = np.random.choice(len(train_datasets), num_train)
    train_datasets = [TensorDataset(t.tensors[0][subset_idxs], t.tensors[1][subset_idxs])\
        for t in train_datasets]

    for model_class, score_func in [(DropoutFCGN(14, 64), mc_dropout_score),
                                    (DropoutFCGN(14, 64), torch.rand_like)]:

        model.backup()
        evaluate_active_learning_performance(model, score_func, deepcopy(train_datasets), deepcopy(test_datasets))
