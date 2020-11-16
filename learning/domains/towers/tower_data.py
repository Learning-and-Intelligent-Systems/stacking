import copy
import numpy as np
import pickle
import torch

from itertools import islice
from torch.utils.data import Dataset, DataLoader, Sampler

from learning.domains.towers.augment_dataset import augment as augment_towers


def preprocess(towers):
    # remove the three color channels at the end of each block encoding
    # (see block_utils.Object.vectorize for details)
    towers = towers[...,:14]
    #towers = towers[...,[0, 1, 2, 4, 5, 7, 8]]
    # convert absolute xy positions to relative positions
    #towers[:,1:,7:9] -= towers[:,:-1,7:9]
    #towers[:,:,1:3] += towers[:,:,7:9]
    towers[:,:,1:4] /= 0.01 #towers[:,:,4:7]
    towers[:,:,7:9] /= 0.01 #towers[:,:,4:6]
    towers[:,:,4:7] = (towers[:,:,4:7] - 0.1) / 0.01
    towers[:,:,0] = (towers[:,:,0] - 0.55)

    return towers.float()

def add_placement_noise(towers):
    for ix in range(towers.shape[0]):
        for bx in range(towers.shape[1]):
            towers[ix, bx, 7:9] += np.random.randn(2)*0.001
    return towers

class TowerDataset(Dataset):
    def __init__(self, tower_dict, K_skip=1, augment=True):
        """ This class assumes the initial dataset contains at least some towers of each size.
        :param tower_dict: A dictionary containing vectorized towers sorted by size.
            {
                '2block' : {
                    'towers': np.array (N, 2, D)
                    'labels': np.array (N,)
                }
                '3block' : ...
            }
        :param K_skip: Option to this the original dataset by taking every K_skip tower. Must be used with augment.
        :param augment: Whether to include augmented towers in the dataset.
        """
        self.tower_keys = ['2block', '3block', '4block', '5block']
        self.tower_tensors = {}
        self.tower_labels = {}

        # First augment the given towers with rotations. 
        if augment:
            augmented_towers = augment_towers(tower_dict, K_skip, False)
        else:
            augmented_towers = tower_dict

        for key in self.tower_keys:
            towers = torch.Tensor(augmented_towers[key]['towers'])
            labels = torch.Tensor(augmented_towers[key]['labels'])

            self.tower_tensors[key] = preprocess(towers)
            self.tower_labels[key] = labels
        
        # Same order as 
        self.start_indices = {}
        self.get_indices()

    def get_indices(self):
        """
        Return a dict of lists. Each sublists has all the indices for towers of a given size.
        """
        indices = {}
        start = 0
        for k in self.tower_keys:
            self.start_indices[k] = start
            indices[k] = np.arange(start, start+self.tower_tensors[k].shape[0])
            start += self.tower_tensors[k].shape[0]
        return indices

    def __getitem__(self, ix):
        """ Translates index to appropriate tower size.
        :param ix: 
        """
        for kx, key in enumerate(self.tower_keys):
            if kx == len(self.tower_keys)-1:
                tower_size = key
                break
            next_key = self.tower_keys[kx+1]
            if ix >= self.start_indices[key] and  ix < self.start_indices[next_key]:
                tower_size = key
                break
        
        tower_ix = ix - self.start_indices[tower_size]
        return self.tower_tensors[tower_size][tower_ix,:,:], self.tower_labels[tower_size][tower_ix]
        
    def __len__(self):
        """
        The total number of towers in the entire dataset.
        """
        return sum(self.tower_tensors[k].shape[0] for k in self.tower_keys)     

    def add_to_dataset(self, tower_dict):
        """
        :param tower_dict: A dictionary of the same format as was passed in initially with
        the towers to add to the dataset.
        """
        augmented_towers = augment_towers(tower_dict, 1, mirror=False)
        for k in self.tower_keys:
            if augmented_towers[k]['towers'].shape[0] > 0:
                new_towers = torch.Tensor(augmented_towers[k]['towers'])
                new_towers = add_placement_noise(new_towers)
                new_towers = preprocess(new_towers)
                new_labels = torch.Tensor(augmented_towers[k]['labels'])
                
                self.tower_tensors[k] = torch.cat([self.tower_tensors[k], new_towers], dim=0)
                self.tower_labels[k] = torch.cat([self.tower_labels[k], new_labels], dim=0)

        self.get_indices()


class TowerSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Build indices for each tower size. 
        indices = self.dataset.get_indices()
        if self.shuffle:
            for k in self.dataset.tower_keys:
                np.random.shuffle(indices[k])

        # Make each list an iterable of batches.
        iterators = {}
        for k in self.dataset.tower_keys:
            arr = iter(indices[k])
            n_batches = (len(indices[k]) + self.batch_size - 1) // self.batch_size
            batches = [list(islice(arr, self.batch_size)) for _ in range(n_batches)]
            iterators[k] = iter(batches)

        # Loop over batches until all the iterators are empty.
        valid_tower_sizes = copy.deepcopy(self.dataset.tower_keys)
        while len(valid_tower_sizes) > 0:
            if self.shuffle:
                key = np.random.choice(valid_tower_sizes)
            else:
                key = valid_tower_sizes[0]
            try:
                yield next(iterators[key])
            except:
                valid_tower_sizes.remove(key)
            
    def __len__(self):
        """ Return the number of batches in the data loader. """
        n_batches = 0
        for k in self.dataset.tower_keys:
            indices = self.dataset.get_indices()
            n_batches += (len(indices[k]) + self.batch_size - 1) // self.batch_size
        return n_batches


if __name__ == '__main__':
    with open('learning/data/random_blocks_(x40000)_5blocks_all.pkl', 'rb') as handle:
        towers_dict = pickle.load(handle)

    dataset = TowerDataset(towers_dict, augment=True, K_skip=10000)
    print('----- Load dataset from file -----')
    print('Num Towers:', len(dataset))
    print('Indices per category:')
    print(dataset.get_indices())
    print('Iterate over dataset:')
    for ix in range(len(dataset)):
        x, y = dataset[ix]
        print(x.shape)

    print('----- Test tower sampler -----')
    sampler = TowerSampler(dataset, 5, True)
    for batch_ixs in sampler:
        print(batch_ixs)

    loader = DataLoader(dataset=dataset,
                        batch_sampler=sampler)
    print(len(loader))

    
