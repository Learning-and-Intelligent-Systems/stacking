import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader

from block_utils import Object
from learning.domains.towers.generate_tower_training_data import sample_random_tower
from learning.domains.towers.tower_data import TowerDataset, TowerSampler
from tower_planner import TowerPlanner

class EnsemblePlanner:
    def __init__(self, logger, n_samples=5000):
        self.tower_keys = ['2block', '3block', '4block', '5block']
        self.n_samples = n_samples
        self.tp = TowerPlanner(stability_mode='contains')
        self.logger = logger

    def generate_candidate_towers(self, blocks, num_blocks=None, discrete=False):
        tower_vectors = []
        tower_block_ids = []
        for _ in range(0, self.n_samples):
            tower, rotated_tower = sample_random_tower(blocks, num_blocks=num_blocks, \
                                        ret_rotated=True, discrete=discrete)
            tower_vectors.append([b.vectorize() for b in rotated_tower])
            tower_block_ids.append([b.get_id() for b in rotated_tower])
        return tower_vectors, tower_block_ids

    def plan(self, blocks, ensemble, reward_fn, args, num_blocks=None, discrete=False):
        #n = len(blocks)
        #max_height = 0
        #max_tower = []

        # Step (1): Build dataset of potential towers. 
        tower_vectors, tower_block_ids = self.generate_candidate_towers(blocks, num_blocks, discrete=discrete)
        
        # Step (2): Get predictions for each tower.
        towers = np.array(tower_vectors)
        block_ids = np.array(tower_block_ids)
        if args.planning_model == 'learned':
            # Since we are only planning for towers of a single size,
            # always use the '2block' key for simplicity. The rest currently
            # need at least some data for the code to work.
            labels = np.zeros((towers.shape[0],))
            tower_dict = {}
            for k in self.tower_keys:
                tower_dict[k] = {}
                if k == '2block':
                    tower_dict[k]['towers'] = towers
                    tower_dict[k]['labels'] = labels
                    tower_dict[k]['block_ids'] = block_ids
                else:
                    tower_dict[k]['towers'] = towers[:5,...]
                    tower_dict[k]['labels'] = labels[:5]
                    tower_dict[k]['block_ids'] = block_ids[:5,...]
    
            tower_dataset = TowerDataset(tower_dict, augment=False)
            tower_sampler = TowerSampler(dataset=tower_dataset,
                                         batch_size=64,
                                         shuffle=False)
            tower_loader = DataLoader(dataset=tower_dataset,
                                      batch_sampler=tower_sampler)
            preds = []
            if hasattr(self.logger.args, 'sampler') and self.logger.args.sampler == 'sequential':
                for tensor, _ in tower_loader:
                    sub_tower_preds = []
                    for n_blocks in range(2, tensor.shape[1]+1):
                        if torch.cuda.is_available():
                            tensor = tensor.cuda()
                        with torch.no_grad():
                            sub_tower_preds.append(ensemble.forward(tensor[:, :n_blocks, :]))
                    sub_tower_preds = torch.stack(sub_tower_preds, dim=0)
                    preds.append(sub_tower_preds.prod(dim=0))
            else:
                for tensor, _ in tower_loader:
                    if torch.cuda.is_available():
                        tensor = tensor.cuda()
                    with torch.no_grad():
                        preds.append(ensemble.forward(tensor))
        
            p_stables = torch.cat(preds, dim=0).mean(dim=1)
            
        elif args.planning_model == 'noisy-model':
            n_estimate = 100
            p_stables = np.zeros(len(tower_vectors))
            for ti, tower_vec in enumerate(tower_vectors):
                # estimate prob of constructability
                results = np.ones(n_estimate)
                for n in range(n_estimate):
                    noisy_tower = []
                    for block_vec in tower_vec:
                        noisy_block = deepcopy(block_vec)
                        noisy_block[7:9] += np.random.randn(2)*args.xy_noise
                        noisy_tower.append(noisy_block)
                        block_tower = [Object.from_vector(block) for block in noisy_tower]
                        if not self.tp.tower_is_constructable(block_tower):
                            results[n] = 0.
                            break
                p_stables[ti] = np.mean(results)
                
        elif args.planning_model == 'simple-model':
            p_stables = np.zeros(len(towers_vectors))
            for ti, tower_vec in enumerate(tower_vectors):
                block_tower = [Object.from_vector(block) for block in tower_vec]
                if self.tp.tower_is_constructable(block_tower):
                    p_stables = 1.

        # Step (3): Find the tallest tower of a given height.
        max_reward, max_exp_reward, max_tower, max_stable = -100, -100, None, 0
        ground_truth = -100
        max_reward_block_ids = None
        for ix, (p, tower, tower_block_ids) in enumerate(zip(p_stables, towers, block_ids)):
            reward = reward_fn(tower)
            exp_reward = p*reward
            if exp_reward >= max_exp_reward:# and p > 0.5:
                if exp_reward > max_exp_reward or (exp_reward == max_exp_reward and p > max_stable):
                    max_tower = tower_vectors[ix]
                    max_reward = reward
                    max_exp_reward = exp_reward
                    max_stable = p
                    max_reward_block_ids = tower_block_ids

            # Check ground truth stability to find maximum reward.
            if self.tp.tower_is_constructable([Object.from_vector(tower[ix,:]) for ix in range(tower.shape[0])]) \
                and reward > ground_truth:
                ground_truth = reward

        if max_tower is None:
            print('None Found')
            max_tower = tower_vectors[0]
            max_reward = reward_fn(towers[0])

        return max_tower, max_reward, ground_truth, max_reward_block_ids
