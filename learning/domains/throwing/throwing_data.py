import numpy as np
import torch
from torch.utils.data import DataLoader

from agents.throwing_agent import ThrowingAgent
from learning.domains.throwing.entities import ThrowingBall, ThrowingAction

def sample_action(obj_ids, n_samples=1):
    """ sample which object to throw, and the parameters of that throw
    """
    z_ids = np.random.choice(a=obj_ids, size=n_samples, replace=True)
    ang = np.random.uniform(np.pi/8, 3*np.pi/8, size=n_samples)
    w = np.random.uniform(-10, 10, size=n_samples)
    return np.stack([ang, w], axis=1), z_ids

def label_actions(objects, actions, z_ids):
    """ produce an outcome for executing an action on an object
    """
    agent = ThrowingAgent(objects)
    ys = []

    for a, z_id in zip(actions, z_ids):
        b = objects[z_id]
        act = ThrowingAction.from_vector(b, a)
        ys.append(agent.run(act))

    return np.array(ys)

def construct_xs(objects, actions, z_ids):
    """ construct a vectorized representation of an action and 
    observable object parameters
    """
    xs = []

    for a, z_id in zip(actions, z_ids):
        b = objects[z_id]
        v = b.vectorize()[[0,1,2,4,5,6,7,8,9]]
        x = np.concatenate([a, v])
        xs.append(x)

    return np.array(xs)

def generate_dataset(objects, n_data, as_tensor=True):
    obj_ids = np.arange(len(objects))
    actions, z_ids = sample_action(obj_ids, n_samples=n_data)
    xs = construct_xs(objects, actions, z_ids)
    ys = label_actions(objects, actions, z_ids)
    dataset = xs, z_ids, ys
    return (torch.Tensor(d) for d in dataset) if as_tensor else dataset

def generate_objects(n_objects):
    return [ThrowingBall.random() for _ in range(n_objects)]


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
        for _ in range(n_dataloaders):
            loader = DataLoader(dataset=dataset,
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