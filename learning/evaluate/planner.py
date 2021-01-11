import numpy as np
import torch

from torch.utils.data import DataLoader

from block_utils import Object
from learning.domains.towers.generate_tower_training_data import sample_random_tower
from learning.domains.towers.tower_data import TowerDataset, TowerSampler
from tower_planner import TowerPlanner

class EnsemblePlanner:
    def __init__(self, n_samples=5000):
        self.tower_keys = ['2block', '3block', '4block', '5block']
        self.n_samples = n_samples
        self.tp = TowerPlanner(stability_mode='contains')

    def generate_candidate_towers(self, blocks, num_blocks=None, discrete=False):
        tower_vectors = []
        for _ in range(0, self.n_samples):
            tower, rotated_tower = sample_random_tower(blocks, num_blocks=num_blocks, \
                                        ret_rotated=True, discrete=discrete)
            tower_vectors.append([b.vectorize() for b in rotated_tower])
        return tower_vectors

    def plan(self, blocks, ensemble, reward_fn, num_blocks=None, discrete=False):
        #n = len(blocks)
        #max_height = 0
        #max_tower = []

        # Step (1): Build dataset of potential towers. 
        tower_vectors = self.generate_candidate_towers(blocks, num_blocks, discrete=discrete)
        
        # Since we are only planning for towers of a single size,
        # always use the '2block' key for simplicity. The rest currently
        # need at least some data for the code to work.
        towers = np.array(tower_vectors)
        labels = np.zeros((towers.shape[0],))
        tower_dict = {}
        for k in self.tower_keys:
            tower_dict[k] = {}
            if k == '2block':
                tower_dict[k]['towers'] = towers
                tower_dict[k]['labels'] = labels
            else:
                tower_dict[k]['towers'] = towers[:5,...]
                tower_dict[k]['labels'] = labels[:5]


        # Step (2): Get predictions for each tower.
        preds = []
        tower_dataset = TowerDataset(tower_dict, augment=False)
        tower_sampler = TowerSampler(dataset=tower_dataset,
                                     batch_size=64,
                                     shuffle=False)
        tower_loader = DataLoader(dataset=tower_dataset,
                                  batch_sampler=tower_sampler)

        for tensor, _ in tower_loader:
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            with torch.no_grad():
                preds.append(ensemble.forward(tensor))

        p_stables = torch.cat(preds, dim=0).mean(dim=1)

        # Step (3): Find the tallest tower of a given height.
        max_reward, max_exp_reward, max_tower, max_stable = -100, -100, None, 0
        ground_truth = -100
        for ix, (p, tower) in enumerate(zip(p_stables, towers)):
            reward = reward_fn(tower)
            exp_reward = p*reward
            if exp_reward >= max_exp_reward and p > 0.5:
                if exp_reward > max_exp_reward or (exp_reward == max_exp_reward and p > max_stable):
                    max_tower = tower_vectors[ix]
                    max_reward = reward
                    max_exp_reward = exp_reward
                    max_stable = p

            # Check ground truth stability to find maximum reward.
            if self.tp.tower_is_constructable([Object.from_vector(tower[ix,:]) for ix in range(tower.shape[0])]) \
                and reward > ground_truth:
                ground_truth = reward

        if max_tower is None:
            print('None Found')
            max_tower = tower_vectors[0]
            max_reward = reward_fn(towers[0])

        return max_tower, max_reward, ground_truth
