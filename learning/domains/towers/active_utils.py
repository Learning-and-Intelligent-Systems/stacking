import numpy as np
import pickle
import torch

from torch.utils.data import DataLoader

from block_utils import Object, get_rotated_block
from learning.domains.towers.generate_tower_training_data import sample_random_tower, vectorize
from learning.domains.towers.tower_data import TowerDataset, TowerSampler
from tower_planner import TowerPlanner


# TODO: Write a version of this function that does pool based active learning from a given file.
def sample_unlabeled_data(n_samples, tx=None, block_set=None, tower_heights=['2', '3', '4', '5']):
    """ Generate n_samples random towers. For now each sample can also have
    random blocks. We should change this later so that the blocks are fixed 
    (i.e., chosen elsewhere) and we only sample the configuration.
    :param n_samples: Number of random towers to consider.
    :param block_set (optional): blocks to use in towers. generate new blocks if None
    :return: Dict containining numpy arrays of the towers sorted by size.
    """
    '''
    # uncomment to use a curriculum
    if tx is None:
        tower_heights=['2']
    else:
        if tx < 25:
            tower_heights=['2']
        elif tx < 50:
            tower_heights=['2', '3']
        elif tx < 75:
            tower_heights=['2','3','4']
        else:
            tower_heights=['2','3','4','5']
    '''
    keys = [th+'block' for th in tower_heights]
    int_tower_heights = [int(th) for th in tower_heights]
    
    # initialize a dictionary of lists to store the generated data
    sampled_towers = {k: {} for k in keys}
    for k in keys:
        sampled_towers[k]['towers'] = []
        sampled_towers[k]['labels'] = []
        if block_set is not None:
            sampled_towers[k]['block_ids'] = []

    # sample random towers and add them to the lists in the dictionary
    for ix in range(n_samples):
        n_blocks = np.random.randint(min(int_tower_heights), max(int_tower_heights)+1)
        # get n_blocks, either from scratch or from the block set
        if block_set is not None: 
            blocks = np.random.choice(block_set, n_blocks, replace=False)
        else:
            blocks = [Object.random(f'obj_{ix}') for ix in range(n_blocks)]
        # sample a new tower
        tower = sample_random_tower(blocks)
        rotated_tower = [get_rotated_block(b) for b in tower]
        # and save that tower in the sampled_towers dict
        sampled_towers['%dblock' % n_blocks]['towers'].append(vectorize(rotated_tower))
        if block_set is not None:
            block_ids = [int(block.name.strip('obj_')) for block in rotated_tower]
            sampled_towers['%dblock' % n_blocks]['block_ids'].append(block_ids)
    
    # convert all the sampled towers to numpy arrays
    for k in keys:
        sampled_towers[k]['towers'] = np.array(sampled_towers[k]['towers'])
        sampled_towers[k]['labels'] = np.zeros((sampled_towers[k]['towers'].shape[0],))
        if block_set is not None:
            sampled_towers[k]['block_ids'] = np.array(sampled_towers[k]['block_ids'])

    return sampled_towers


def get_predictions(dataset, ensemble):
    """
    :param dataset: A tower_dict structure.
    :param ensemble: The Ensemble model which to use for predictions.
    :return: Return (N, K) array of flat predictions. Predictions are 
    ordered by tower size.
    """
    preds = []
    # Create TowerDataset object.
    tower_dataset = TowerDataset(dataset, augment=False)
    tower_sampler = TowerSampler(dataset=tower_dataset,
                                 batch_size=64,
                                 shuffle=False)
    tower_loader = DataLoader(dataset=tower_dataset,
                              batch_sampler=tower_sampler)

    # Iterate through dataset, getting predictions for each.
    for tensor, _ in tower_loader:
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            preds.append(ensemble.forward(tensor))
        
    return torch.cat(preds, dim=0)


def get_labels(samples):
    """ Takes as input a dictionary from the get_subset function. 
    Augment it with stability labels. 
    :param samples:
    :return:
    """
    tp = TowerPlanner(stability_mode='contains')
    for k in samples.keys():
        if len(samples[k]['towers'].shape) > 1:
            n_towers, n_blocks, _ = samples[k]['towers'].shape
            labels = np.ones((n_towers,))

            for ix in range(0, n_towers):
                # Convert tower to Block representation.
                block_tower = [Object.from_vector(samples[k]['towers'][ix, jx, :]) for jx in range(n_blocks)]
                #  Use tp to check for stability.
                if not tp.tower_is_constructable(block_tower):
                    labels[ix] = 0.

            samples[k]['labels'] = labels
        else:
            samples[k]['labels'] = np.empty(0)
    return samples


def get_subset(samples, indices):
    """ Given a tower_dict structure and indices that are flat,
    return a tower_dict structure with only those indices.
    :param samples: A tower_dict structure.
    :param indices: Which indices of the original structure to select.
    """
    keys = list(samples.keys())
    selected_towers = {k: {'towers': []} for k in keys}
    
    # Initialize tower ranges.
    start = 0
    for k in keys:
        end = start + samples[k]['towers'].shape[0]
        tower_ixs = indices[np.logical_and(indices >= start,
                                        indices < end)] - start
        selected_towers[k]['towers'] = samples[k]['towers'][tower_ixs,...]
        start = end

    return selected_towers


class PoolSampler:

    def __init__(self, pool_fname):
        self.pool_fname = pool_fname
        self.keys = ['2block', '3block', '4block', '5block']
        with open(self.pool_fname, 'rb') as handle:
            self.pool = pickle.load(handle)

    def sample_unlabeled_data(self, n_samples):
        """
        Return all examples that haven't been chosen so far.
        """
        return self.pool

    def get_subset(self, samples, indices):
        """
        Remove chosen examples from the pool.
        """
        selected_towers = {k: {'towers': []} for k in self.keys}

        start = 0
        for k in self.keys:
            end = start + self.pool[k]['towers'].shape[0]
            
            tower_ixs = indices[np.logical_and(indices >= start,
                                        indices < end)] - start
            selected_towers[k]['towers'] = self.pool[k]['towers'][tower_ixs,...]
 
            mask = np.ones(self.pool[k]['towers'].shape[0], dtype=bool)

            mask[tower_ixs] = False
            self.pool[k]['towers'] = self.pool[k]['towers'][mask,...]
            self.pool[k]['labels'] = self.pool[k]['labels'][mask,...]
            
            start = end
        
        return selected_towers
        


if __name__ == '__main__':
    data = sample_unlabeled_data(1000)
    for k in data.keys():
        print(data[k]['towers'].shape)

    indices = np.random.randint(0, 1000, 10)
    selected_towers = get_subset(data, indices)
    print(indices)
    for k in selected_towers.keys():
        print(selected_towers[k]['towers'].shape)

    labeled_towers = get_labels(selected_towers)
    for k in labeled_towers.keys():
        print(labeled_towers[k]['labels'])

    print('----- Test adding new data to dataset -----')
    with open('learning/data/random_blocks_(x40000)_5blocks_all.pkl', 'rb') as handle:
        towers_dict = pickle.load(handle)

    dataset = TowerDataset(towers_dict, augment=True, K_skip=10000)
    sampler = TowerSampler(dataset, 10, False)
    print('----- Initial batches -----')
    for batch_ixs in sampler:
        print(batch_ixs)

    loader = DataLoader(dataset=dataset,
                        batch_sampler=sampler)

    print('Num Initial Towers:', len(dataset))
    print('Initial indices per category:')
    print(dataset.get_indices())

    dataset.add_to_dataset(labeled_towers)
    print('Num Updated Towers:', len(dataset))
    print('Updated indices per category:')
    print(dataset.get_indices())

    print('----- Updated batches -----')
    for batch_ixs in sampler:
        print(batch_ixs)

    
    # print(len(loader))

    print('----- Pool Sampler Test -----')
    sampler = PoolSampler('learning/data/random_blocks_(x40000)_5blocks_uniform_mass.pkl')
    pool = sampler.sample_unlabeled_data(10)
    for k in sampler.keys:
        print(pool[k]['towers'].shape) 

    sampler.get_subset(np.array([0, 1, 2, 3, 4, 20000, 20005]))
    for k in sampler.keys:
        print(pool[k]['towers'].shape) 