import numpy as np
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


def update_params(model, latents, data):
    return model

def update_latents(model, latents, data):
    return latents

def train(model, latents, data, n_epochs=10, n_steps_per_epoch=10):
    for epoch_idx in range(n_epochs):
        for step_idx in range(n_steps_per_epoch):
            latents = update_latents(model, latents, data)
            model = update_params(model, latents, data)

def test(model, latents, data):
    latents = update_latents(model, latents, data)


if __name__ == "__main__":
    # load data
    with open("learning/data/10block_set_(x10000).pkl", 'rb') as handle:
        train_towers_dict = pickle.load(handle)
    train_dataset = TowerDataset(train_towers_dict, augment=True)
    with open("learning/data/10block_set_(x1000).pkl", 'rb') as handle:
        test_towers_dict = pickle.load(handle)
    test_dataset = TowerDataset(test_towers_dict, augment=True)

    # create the model
    model = None

    # create the latents
    latents = None

    # train
    train(model, latents, train_dataset)

    # test
    test(model, latents, test_dataset)