import numpy as np
import pickle
import time
import torch

from copy import deepcopy
from torch.utils.data import DataLoader

from block_utils import World, Environment, Object, Quaternion, Pose, get_rotated_block, ZERO_POS
from learning.domains.towers.generate_tower_training_data import sample_random_tower, vectorize, QUATERNIONS
from learning.domains.towers.tower_data import TowerDataset, TowerSampler, unprocess
from tower_planner import TowerPlanner


def sample_sequential_data(block_set, dataset, n_samples):
    """ Generate n_samples random towers. Each tower has the property that its
    base (the tower until the last block) is stable. To ensure this, we start
    with all the stable towers plus base cases of single block towers.
    :param block_set: List of blocks that can be used in the towers.
    :param dataset: List of current towers that have been built.
    :param n_samples: Number of random towers to consider.
    :param max_blocks
    :return: Dict containining numpy arrays of the towers sorted by size.
    """
    print('Generating data sequentially...')
    keys = ['2block', '3block', '4block', '5block']

    # initialize a dictionary of lists to store the generated data
    sampled_towers = {k: {} for k in keys}
    for k in keys:
        sampled_towers[k]['towers'] = []
        sampled_towers[k]['labels'] = []
        if block_set is not None:
            sampled_towers[k]['block_ids'] = []

    # Gather list of all stable towers. Towers should be of blocks that are rotated in "Block" format.
    stable_towers = []
    # Get all single block stable towers.
    for block in block_set:
        for orn in QUATERNIONS:
            new_block = deepcopy(block)
            new_block.pose = Pose(ZERO_POS, orn)
            rot_block = get_rotated_block(new_block)
            rot_block.pose = Pose((0., 0., rot_block.dimensions.z/2.), (0, 0, 0, 1))
            stable_towers.append([rot_block])

    # Get all stable towers from the dataset.
    for k in keys[:3]:
        if dataset is None:
            break
        tower_tensors = unprocess(dataset.tower_tensors[k].cpu().numpy().copy())
        tower_labels = dataset.tower_labels[k]
        for ix, (tower_vec, tower_label) in enumerate(zip(tower_tensors, tower_labels)):
            if tower_label == 1:
                block_tower = []
                for bx in range(tower_vec.shape[0]):
                    block = Object.from_vector(tower_vec[bx, :])
                    if block_set is not None:
                        block.name = 'obj_'+str(int(dataset.tower_block_ids[k][ix, bx]))
                    block_tower.append(block)
                stable_towers.append(block_tower)

    block_lookup = {}
    for block in block_set:
        block_lookup[block.mass] = block

    # Sample random towers by randomly choosing a stable base then trying to add a block.
    for ix in range(n_samples):
        # Choose a stable base.
        tower_ix = np.random.choice(np.arange(0, len(stable_towers)))
        base_tower = stable_towers[tower_ix]

        # Choose a block that's not already in the tower.
        remaining_blocks = {}
        for k in block_lookup:
            used = False
            for block in base_tower:
                if np.abs(k - block.mass) < 0.0001:
                    used = True
            if not used:
                remaining_blocks[k] = block_lookup[k]
        assert(len(remaining_blocks) == len(block_set) - len(base_tower))

        new_block = deepcopy(np.random.choice(list(remaining_blocks.values())))
        
        # Choose an orientation.
        orn = QUATERNIONS[np.random.choice(np.arange(0, len(QUATERNIONS)))]
        new_block.pose = Pose(ZERO_POS, orn)
        rot_block = get_rotated_block(new_block)
        
        # Sample a displacement.
        base_dims = np.array(base_tower[-1].dimensions)[:2]
        new_dims = np.array(rot_block.dimensions)[:2]
        max_displacements_xy = (base_dims+new_dims)/2.
        noise_xy = np.clip(0.5*np.random.randn(2), -0.95, 0.95)
        rel_xy = max_displacements_xy*noise_xy

        # Calculate the new pose.
        base_pos = np.array(base_tower[-1].pose.pos)[:2]
        pos_xy = base_pos + rel_xy
        pos_z = np.sum([b.dimensions.z for b in base_tower]) + rot_block.dimensions.z/2.
        rot_block.pose = Pose((pos_xy[0], pos_xy[1], pos_z), (0, 0, 0, 1))
        
        # Add block to tower.
        new_tower = base_tower + [rot_block]

        if False:
            w = World(new_tower)
            env = Environment([w], vis_sim=True, vis_frames=True)
            for tx in range(240):
                env.step(vis_frames=True)
                time.sleep(1/240.)
            env.disconnect()
        # Save that tower in the sampled_towers dict
        n_blocks = len(new_tower)
        sampled_towers['%dblock' % n_blocks]['towers'].append(vectorize(new_tower))
    
        # save block id
        if block_set is not None:
            block_ids = [int(block.name.strip('obj_')) for block in new_tower]
            sampled_towers['%dblock' % n_blocks]['block_ids'].append(block_ids)
    
    # convert all the sampled towers to numpy arrays
    for k in keys:
        sampled_towers[k]['towers'] = np.array(sampled_towers[k]['towers'])
        if sampled_towers[k]['towers'].shape[0] == 0:
            sampled_towers[k]['towers'] = sampled_towers[k]['towers'].reshape((0, int(k[0]), 17))
        sampled_towers[k]['labels'] = np.zeros((sampled_towers[k]['towers'].shape[0],))
        if block_set is not None:
            sampled_towers[k]['block_ids'] = np.array(sampled_towers[k]['block_ids'])
    return sampled_towers

def sample_unlabeled_data(n_samples, block_set=None):
    """ Generate n_samples random towers. For now each sample can also have
    random blocks. We should change this later so that the blocks are fixed 
    (i.e., chosen elsewhere) and we only sample the configuration.
    :param n_samples: Number of random towers to consider.
    :param block_set (optional): blocks to use in towers. generate new blocks if None
    :return: Dict containining numpy arrays of the towers sorted by size.
    """
    keys = ['2block', '3block', '4block', '5block']

    # initialize a dictionary of lists to store the generated data
    sampled_towers = {k: {} for k in keys}
    for k in keys:
        sampled_towers[k]['towers'] = []
        sampled_towers[k]['labels'] = []
        if block_set is not None:
            sampled_towers[k]['block_ids'] = []

    # sample random towers and add them to the lists in the dictionary
    for ix in range(n_samples):
        max_blocks = min(6, len(block_set))
        n_blocks = np.random.randint(2, max_blocks)
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

def get_sequential_predictions(dataset, ensemble):
    """
    Make a separate prediction for each of the sub-towers in a tower.
    Return stable only if all sub-towers are stable. This is for the
    model that assumes the base of each tower is stable.
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
        sub_tower_preds = []
        #print(tensor.shape)
        for n_blocks in range(2, tensor.shape[1]+1):
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            with torch.no_grad():
                sub_tower_preds.append(ensemble.forward(tensor[:, :n_blocks, :]))
        sub_tower_preds = torch.stack(sub_tower_preds, dim=0)
        #print('SubTowerPreds:', sub_tower_preds.shape)
        #preds.append(sub_tower_preds[-1,:,:])
        #preds.append(sub_tower_preds.prod(dim=0))
        preds.append((sub_tower_preds > 0.5).all(dim=0).float())
        #print(preds[-1].shape)
    return torch.cat(preds, dim=0)

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


def get_labels(samples, exec_mode, agent, xy_noise=0.003):
    """ Takes as input a dictionary from the get_subset function. 
    Augment it with stability labels. 
    :param samples:
    :param exec_mode: str in ['simple-model', 'noisy-model', 'sim', 'real']
    :param agent: PandaAgent or None (if exec_mode == 'simple-model' or 'noisy-model')
    :return:
    """
    tp = TowerPlanner(stability_mode='contains')
    for k in samples.keys():
        n_towers, n_blocks, _ = samples[k]['towers'].shape
        labels = np.ones((n_towers,))

        for ix in range(0, n_towers):
            # Add noise to blocks and convert tower to Block representation.
            block_tower = []
            for jx in range(n_blocks): 
                vec_block = samples[k]['towers'][ix, jx, :]
                if exec_mode == 'noisy-model':
                    vec_block[7:9] += np.random.randn(2)*xy_noise
                block = Object.from_vector(vec_block) # block is already rotated
                if 'block_ids' in samples[k].keys():
                    block.name = 'obj_'+str(samples[k]['block_ids'][ix, jx])
                block_tower.append(block)
            #  Use tp to check for stability.
            if exec_mode == 'simple-model' or exec_mode == 'noisy-model':
                if not tp.tower_is_constructable(block_tower):
                    labels[ix] = 0.
            else:
                vis = True
                success = False
                real = exec_mode == 'real'
                # if planning fails, reset and try again
                while not success:
                    if agent.use_action_server:
                        success, label = agent.simulate_tower_parallel(block_tower, vis, real=real)
                    else:
                        success, label = agent.simulate_tower(block_tower, vis, real=real)
                    if not success:
                        if real:
                            input('Resolve conflict causing planning to fail, then press \
                                    enter to try again.')
                        else: # in sim
                            input('Should reset sim. Not yet handled. Exit and restart training.')
                labels[ix] = label
        samples[k]['labels'] = labels
    return samples


def get_subset(samples, indices):
    """ Given a tower_dict structure and indices that are flat,
    return a tower_dict structure with only those indices.
    :param samples: A tower_dict structure.
    :param indices: Which indices of the original structure to select.
    """
    keys = ['2block', '3block', '4block', '5block']
    selected_towers = {k: {'towers': [], 'block_ids': []} for k in keys}
    
    # Initialize tower ranges.
    start = 0
    for k in keys:
        end = start + samples[k]['towers'].shape[0]
        tower_ixs = indices[np.logical_and(indices >= start,
                                        indices < end)] - start
        selected_towers[k]['towers'] = samples[k]['towers'][tower_ixs,...]
        if 'block_ids' in selected_towers[k].keys():
            selected_towers[k]['block_ids'] = samples[k]['block_ids'][tower_ixs,...]
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

    # print('----- Pool Sampler Test -----')
    # sampler = PoolSampler('learning/data/random_blocks_(x40000)_5blocks_uniform_mass.pkl')
    # pool = sampler.sample_unlabeled_data(10)
    # for k in sampler.keys:
    #     print(pool[k]['towers'].shape) 

    # sampler.get_subset(np.array([0, 1, 2, 3, 4, 20000, 20005]))
    # for k in sampler.keys:
    #     print(pool[k]['towers'].shape) 


    print('----- Sequential Sampler Test -----')
    with open('learning/data/block_set_10.pkl', 'rb') as handle:
        block_set = pickle.load(handle)

    with open('learning/experiments/logs/towers-con-init-random-blocks-10-fcgn-f1val-100k-20201213-152931/datasets/active_10.pkl', 'rb') as handle:
        dataset = pickle.load(handle)

    data_sampler_fn = lambda n_samples: sample_sequential_data(block_set, dataset, n_samples)
    unlabeled = data_sampler_fn(10000)