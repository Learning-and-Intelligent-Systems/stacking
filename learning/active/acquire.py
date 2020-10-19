import numpy as np
import torch

from learning.active.toy_data import ToyDataGenerator, ToyDataset
from learning.active.utils import get_predictions


def bald(predictions, eps=1e-5):
    """ Get the BALD score for each example.
    :param predictions: (N, K) predictions for N datapoints from K models.
    :return: (N,) The BALD score for each of the datapoints.
    """
    mp_c1 = torch.mean(predictions, dim=1)
    mp_c0 = torch.mean(1 - predictions, dim=1)

    m_ent = -(mp_c1 * torch.log(mp_c1+eps) + mp_c0 * torch.log(mp_c0+eps))

    p_c1 = predictions
    p_c0 = 1 - predictions
    ent_per_model = p_c1 * torch.log(p_c1+eps) + p_c0 * torch.log(p_c0+eps)
    ent = torch.mean(ent_per_model, dim=1)

    bald = m_ent + ent
    return bald

def sample_unlabeled_data(n_samples):
    """ Randomly sample datapoints without labels. 
    :param n_samples: The number of samples to return.
    :return: np.array(n_samples, 2)
    """
    gen = ToyDataGenerator()
    xs, _ = gen.generate_uniform_dataset(n_samples)
    return xs

def choose_acquisition_data(samples, ensemble, n_acquire):
    """ Choose data points with the highest acquisition score
    :param samples: (N,2) An array of unlabelled datapoints which to evaluate.
    :param ensemble: A list of models. 
    :param n_acquire: The number of data points to acquire.
    :return: (n_acquire, 2) - the samples which to label.
    """
    # Get predictions for each model of the ensemble. Note these ys won't be used.
    placeholder_ys = np.zeros((samples.shape[0],), dtype='float32')
    dataset = ToyDataset(samples, placeholder_ys)
    preds = get_predictions(dataset, ensemble)

    # Get the BALD score for each.
    scores = bald(preds).numpy()
    print(scores.shape)

    # Return the n_acquire points with the highest score.
    acquire_indices = np.argsort(scores)[::-1][:n_acquire]
    return samples[acquire_indices, :]

def get_labels(samples):
    """ Get the labels for the chosen datapoints.
    :param samples: (n_acquire, 2)
    :return: (n_acquire,) The labels for the given datapoints.
    """
    gen = ToyDataGenerator()
    ys = gen.get_labels(samples)
    return ys

def acquire_datapoints(ensemble, n_samples, n_acquire):
    """ Get new datapoints given the current set of models.
    Calls the next three methods in turn with their respective 
    parameters.
    :return: (n_acquire, 2), (n_acquire,) - x,y tuples of the new datapoints.
    """
    unlabeled_pool = sample_unlabeled_data(n_samples)
    xs = choose_acquisition_data(unlabeled_pool, ensemble, n_acquire)
    ys = get_labels(xs)
    return xs, ys
