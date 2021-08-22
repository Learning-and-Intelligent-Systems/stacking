import numpy as np
import torch
from torch.utils.data import DataLoader

from agents.throwing_agent import ThrowingAgent
from learning.domains.throwing.entities import ThrowingBall, ThrowingAction

def sample_actions(obj_ids, n_samples=1):
    """ sample which object to throw, and the parameters of that throw
    """
    z_ids = np.random.choice(a=obj_ids, size=n_samples, replace=True)
    return ThrowingAction.random_vector(n_samples=n_samples), z_ids

def label_actions(objects, actions, z_ids, as_tensor=False):
    """ produce an outcome for executing an action on an object
    """
    agent = ThrowingAgent(objects)
    ys = []

    for a, z_id in zip(actions, z_ids):
        b = objects[int(z_id)]
        act = ThrowingAction.from_vector(b, a)
        ys.append(agent.run(act))

    return torch.Tensor(ys) if as_tensor else np.array(ys)

def construct_xs(objects, actions, z_ids):
    """ construct a vectorized representation of an action and
    observable object parameters
    """
    xs = []

    for a, z_id in zip(actions, z_ids):
        b = objects[z_id]
        v = b.vectorize()
        x = np.concatenate([a, v])
        xs.append(x)

    return np.array(xs)

def xs_to_actions(xs):
    return xs[:,:2]

def generate_dataset(objects,
                     n_data,
                     as_tensor=True,
                     label=True):
    obj_ids = np.arange(len(objects))
    actions, z_ids = sample_actions(obj_ids, n_samples=n_data)
    xs = construct_xs(objects, actions, z_ids)
    if label:
        ys = label_actions(objects, actions, z_ids)
        dataset = xs, z_ids, ys
    else:
        dataset = xs, z_ids
    return tuple(torch.Tensor(d) for d in dataset) if as_tensor else dataset

def generate_objects(n_objects):
    return [ThrowingBall.random() for _ in range(n_objects)]

def make_x_partially_observable(xs, hide_dims):
    fully_obs_dim = ThrowingBall.dim + ThrowingAction.dim
    keep_dims = list(set(np.arange(fully_obs_dim)).difference(set(hide_dims)))
    return xs[..., keep_dims]

class ParallelDataLoader:
    def __init__(self, dataset, batch_size, shuffle, n_dataloaders=1):
        """ Wrap multiple dataloaders so that we iterate through the data independently and in parallel.
        :param dataset: The underlying dataset to iterate over.
        :param batch_size: Batch size.
        :param shuffle: Whether to shuffle the data.
        :param n_dataloaders: The number of underlying independent dataloaders to use.
        """
        # Create a custom sampler and loader so each loader uses idependently shuffled data.
        self.loaders = []
        self.dataset = dataset
        for _ in range(n_dataloaders):
            loader = DataLoader(dataset=self.dataset,
                                batch_size=batch_size,
                                shuffle=shuffle)
            self.loaders.append(loader)

    def __iter__(self):
        # Initialize the dataloaders (this should reshuffle the data in each).
        loaders = [iter(l) for l in self.loaders]
        stop = False
        # Return a separate batch for each loader at each step.
        while not stop:
            batches = []
            for loader in loaders:
                try:
                    batches.append(next(loader))
                except:
                    # print('[ParallelDataLoader] Warning: failed to get batch from all loaders.')
                    stop = True
            if not stop:
                yield batches

    def __len__(self):
        return len(self.loaders[0])

    def add(self, *data):
        n = len(self.dataset.tensors)
        assert n == len(data), "Require same number of tensors"
        self.dataset.tensors = tuple(torch.cat([self.dataset.tensors[i], data[i]], axis=0) for i in range(n))

