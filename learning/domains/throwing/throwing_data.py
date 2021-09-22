from multiprocessing import Process
import numpy as np
from tqdm.contrib import tzip
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

def sample_same_actions(obj_ids, n_samples=1):
    """ sample which object to throw, and the parameters of that throw
    """
    samples_per_obj = n_samples//len(obj_ids)
    base_actions = ThrowingAction.random_vector(n_samples=samples_per_obj)
    z_ids = np.zeros((n_samples,), dtype=np.int16)
    actions = np.zeros((n_samples, 2))
    for ix in range(0, len(obj_ids)):
        actions[ix*samples_per_obj:(ix+1)*samples_per_obj, :] = base_actions
        z_ids[ix*samples_per_obj:(ix+1)*samples_per_obj] = ix
    
    return actions, z_ids

def label_actions(objects, actions, z_ids, as_tensor=False):
    """ produce an outcome for executing an action on an object
    """
    agent = ThrowingAgent(objects)
    ys = []

    for a, z_id in tzip(actions, z_ids): # tzip for progress bar 
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
    return xs[:,-2:]

def generate_dataset(objects,
                     n_data,
                     as_tensor=True,
                     label=True,
                     duplicate=False):
    obj_ids = np.arange(len(objects))
    if not duplicate:
        actions, z_ids = sample_actions(obj_ids, n_samples=n_data)
    else:
        actions, z_ids = sample_same_actions(obj_ids, n_samples=n_data)
    xs = construct_xs(objects, actions, z_ids)
    if label:
        ys = label_actions(objects, actions, z_ids)
        dataset = xs, z_ids, ys
    else:
        dataset = xs, z_ids
    return tuple(torch.Tensor(d) for d in dataset) if as_tensor else dataset

def generate_dataset_with_repeated_actions(objects,
                                           n_actions,
                                           as_tensor=True,
                                           label=True):
    """ same as generate_dataset, but all the actions are the same for each object """
    n_objects = len(objects)
    base_z_ids = np.arange(n_objects)
    base_actions = ThrowingAction.random_vector(n_samples=n_actions)
    actions = np.vstack([np.meshgrid(acts, base_z_ids)[0].ravel() for acts in base_actions.T]).T
    z_ids = np.meshgrid(base_actions[:,0], base_z_ids)[1].ravel().astype(int)
    print(actions)
    print(z_ids)
    xs = construct_xs(objects, actions, z_ids)
    if label:
        ys = label_actions(objects, actions, z_ids)
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

def parse_hide_dims(hide_dims):
    return [int(d) for d in hide_dims.split(',')] if hide_dims != "" else []

def make_x_partially_observable(xs, hide_dims):
    # the fully observed dimension is the ball parameters and the throw parameters
    fully_obs_dim = ThrowingBall.dim + ThrowingAction.dim
    keep_dims = list(set(np.arange(fully_obs_dim)).difference(set(hide_dims)))
    return xs[..., keep_dims]

def _normalize_x(x):
    x_mean = torch.Tensor([ 1.0000000e+00,   # R
                            0.0000000e+00,   # G
                            0.0000000e+00,   # B
                            1.0000000e+00,   # mass
                            4.0504143e-02,   # radius
                            1.0000000e+00,   # drag linear
                            1.0000000e-05,   # drag angular
                            8.0000000e-01,   # friction
                            1.0000000e-04,   # rolling resistance
                            4.8150727e-01,   # bounciness
                            7.8539816e-01,   # release angle
                            0.0000000e+00])  # release spin

    x_var = torch.Tensor([  0.0000000e+00,   # R
                            0.0000000e+00,   # G
                            0.0000000e+00,   # B
                            0.0000000e+00,   # mass
                            1.2525999e-04,   # radius
                            0.0000000e+00,   # drag linear
                            0.0000000e+00,   # drag angular
                            0.0000000e+00,   # friction
                            0.0000000e+00,   # rolling resista
                            3.6030598e-02,   # bounciness
                            5.2659944e-02,   # release angle
                            3.3095131e+01])  # release spin

    return (x - x_mean)/torch.maximum(torch.sqrt(x_var), torch.Tensor([1e-6]))

def _normalize_y(y):
    y_mean = 0.9944
    y_var = 0.2188
    return (y - y_mean)/np.sqrt(y_var)

def _unnormalize_y(y, use_mean=True):
    y_mean = 0.9944 * use_mean
    y_var = 0.2188
    return y * np.sqrt(y_var) + y_mean


def preprocess_batch(batch, hide_dims, normalize_x, normalize_y):
    """ batch preprocessing before feeding into the NN """
    x = batch[0]
    z_id = batch[1]
    y = batch[2] if len(batch) == 3 else None

    if torch.cuda.is_available():
        x = x.cuda()
        z_id = z_id.cuda()
        if y is not None:
            y = y.cuda()

    if normalize_x:
        x = _normalize_x(x)
    if normalize_y and y is not None:
        y = _normalize_y(y)

    x = make_x_partially_observable(x, hide_dims)

    if y is not None:
        return x, z_id, y
    else:
        return x, z_id

def postprocess_pred(pred, unnormalize):
    D_pred = pred.shape[-1] // 2
    if len(pred.shape) > 3:
        pred = torch.flatten(pred, start_dim=1, end_dim=2) # if neither dim is marginalized, flatten them
    mu, log_sigma = torch.split(pred, D_pred, dim=-1)
    sigma = 0.1 + torch.exp(log_sigma)
    if unnormalize:
        return _unnormalize_y(mu), _unnormalize_y(sigma, use_mean=False)
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
