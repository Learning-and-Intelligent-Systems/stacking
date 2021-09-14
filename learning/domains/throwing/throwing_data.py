from multiprocessing import Process
import numpy as np
import torch
from torch.utils.data import DataLoader

from agents.throwing_agent import ThrowingAgent
from learning.domains.throwing.entities import ThrowingBall, ThrowingAction

###############################################################################
# dataset generation
###############################################################################

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
        x = np.concatenate([v, a])
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
        ys = label_actions(objects, actions, z_ids, as_tensor=as_tensor)
        dataset = xs, z_ids, ys
    else:
        dataset = xs, z_ids
    return tuple(torch.Tensor(d) for d in dataset) if as_tensor else dataset

def generate_dataset_parallel(objects,
                              n_data,
                              as_tensor=True,
                              label=True,
                              n_workers=8):

    n_data_per_process = int(np.ceil(n_data/n_workers))
    processes = [Process(target=generate_dataset,
                         args=(objects, n_data_per_process),
                         kwargs={"as_tensor": as_tensor, "label": label}) for _ in range(n_workers)]
    for p in processes:
        p.start()

    results = []
    for p in processes:
        # TODO(izzy): figure out how to get the result back from the children
        results.append(p.join())

    return tuple(torch.cat(d, axis=0) for d in zip(results))

def generate_objects(n_objects):
    return [ThrowingBall.random() for _ in range(n_objects)]

###############################################################################
# data pre/post processing for NN training
###############################################################################

def make_x_partially_observable(xs, hide_dims):
    # the fully observed dimension is the ball parameters and the throw parameters
    fully_obs_dim = ThrowingBall.dim + ThrowingAction.dim
    keep_dims = list(set(np.arange(fully_obs_dim)).difference(set(hide_dims)))
    return xs[..., keep_dims]

def normalize_x(x):
    x_mean = torch.Tensor([ 1.,  0.,  0.,  1.,
        4.0470269e-02,  1.,  1e-5,  0.8,
        1e-4,  4.3595454e-01,  7.8840643e-01, -3.1721351e-01])
    x_var = torch.Tensor([0., 0., 0., 0.,
       1.2081314e-04, 0., 0., 0.,
       0., 4.1699380e-02, 5.3409986e-02, 3.3062962e+01])

    return (x - x_mean)/torch.sqrt(x_var + 1e-6)

def normalize_y(y):
    y_mean = 1.0001804
    y_var = 0.21319202
    return (y - y_mean)/np.sqrt(y_var)

def unnormalize_y(y):
    y_mean = 1.0001804
    y_var = 0.21319202
    return y * np.sqrt(y_var) + y_mean

def preprocess_batch(batch, hide_dims, normalize):
    """ batch preprocessing before feeding into the NN """
    x = batch[0]
    z_id = batch[1]
    y = batch[2] if len(batch) == 3 else None

    if torch.cuda.is_available():
        x = x.cuda()
        z_id = z_id.cuda()
        if y is not None:
            y = y.cuda()

    if normalize:
        x = normalize_x(x)
        if y is not None:
            y = normalize_y(y)

    x = make_x_partially_observable(x, hide_dims)

    if y is not None:
        return x, z_id, y
    else:
        return x, z_id

def postprocess_pred(pred, unnormalize):
    D_pred = pred.shape[-1] // 2
    mu, log_sigma = torch.split(pred, D_pred, dim=-1)
    sigma = torch.exp(log_sigma)
    if unnormalize:
        return unnormalize_y(mu), unnormalize_y(sigma)
    else:
        return mu, sigma

###############################################################################
# dataloader for training ensembles
###############################################################################

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


if __name__ == '__main__':
    objects = generate_objects(10)
    dataset = generate_dataset_parallel(objects, 30)
