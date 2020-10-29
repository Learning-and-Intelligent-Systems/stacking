import numpy as np
import torch

from torch.utils.data import DataLoader

from learning.domains.toy2d.toy_data import ToyDataGenerator, ToyDataset


def sample_unlabeled_data(n_samples):
    """ Randomly sample datapoints without labels. 
    :param n_samples: The number of samples to return.
    :return: np.array(n_samples, 2)
    """
    gen = ToyDataGenerator()
    xs, _ = gen.generate_uniform_dataset(n_samples)
    return xs


def get_predictions(dataset, ensemble):
    """
    :param dataset: Data returned from sample_unlabeled_data.
    :param ensemble: An ensemble of K models on which to get predictions.
    :return: A (NxK) array with classification probabilities for each model.
    """
    if type(dataset) != ToyDataset:
        placeholder_ys = np.zeros((dataset.shape[0],), dtype='float32')
        dataset = ToyDataset(dataset, placeholder_ys)
    loader = DataLoader(dataset, shuffle=False, batch_size=32)

    preds = []
    for tensor, _ in loader:
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            preds.append(ensemble.forward(tensor))
        
    return torch.cat(preds, dim=0)


def get_labels(samples):
    """ Get the labels for the chosen datapoints.
    :param samples: (n_acquire, 2)
    :return: (n_acquire,) The labels for the given datapoints.
    """
    gen = ToyDataGenerator()
    ys = gen.get_labels(samples)
    return ys


def get_subset(samples, indices):
    return samples[indices, :]