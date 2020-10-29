import numpy as np

from block_utils import Object, get_rotated_block
from learning.domains.towers.generate_tower_training_data import sample_random_tower, vectorize


def sample_unlabeled_data(n_samples):
    """ Generate n_samples random towers. For now each sample can also have
    random blocks. We should change this later so that the blocks are fixed 
    (i.e., chosen elsewhere) and we only sample the configuration.
    :param n_samples: Number of random towers to consider.
    :return: Dict containining numpy arrays of the towers sorted by size.
    """
    keys = ['2block', '3block', '4block', '5block']
    sampled_towers = {k: {'towers': []} for k in keys}

    for ix in range(n_samples):
        n_blocks = np.random.randint(2, 6)
        blocks = [Object.random(f'obj_{ix}') for ix in range(n_blocks)]
        tower = sample_random_tower(blocks)
        rotated_tower = [get_rotated_block(b) for b in tower]

        sampled_towers['%dblock' % n_blocks]['towers'].append(vectorize(rotated_tower))
    
    for k in keys:
        sampled_towers[k]['towers'] = np.array(sampled_towers[k]['towers'])

    return sampled_towers

def get_predictions(dataset, ensemble):
    """
    :param dataset: Return (N, K) array of flat predictions.
    """
    pass

def get_labels(samples):
    """ Takes as input a dictionary from the sampled_unlabeled_data 

    """
    pass

def get_subset(samples, indices):
    pass


if __name__ == '__main__':
    data = sample_unlabeled_data(1000)
    for k in data.keys():
        print(data[k]['towers'].shape)