import numpy as np
import torch
from copy import deepcopy
import pickle
import os
from torch.utils.data import DataLoader

from block_utils import Object, World, Environment, get_rotated_block
from learning.domains.towers.generate_tower_training_data import sample_random_tower
from learning.domains.towers.tower_data import TowerDataset, TowerSampler
from tower_planner import TowerPlanner

class EnsemblePlanner:
    def __init__(self, logger, n_samples=None):
        self.tower_keys = ['2block', '3block', '4block', '5block']
        self.n_samples = {2: 5000, 3: 10000, 4: 20000,  5: 100000, 6: 250000, 7:500000}
        self.tp = TowerPlanner(stability_mode='contains')
        self.logger = logger
        self.using_cache = False

    def get_cached_towers(self, args, num_blocks, blocks, n_tower):
        key = '%dblock' % num_blocks
        if not self.using_cache:
            if args.block_set_fname == '':
                cache_name = 'random_set.pkl'
            else:
                cache_name = os.path.basename(args.block_set_fname)
            
            # see if cache for this block set exists    
            try:
                with open(os.path.join('learning/evaluate/cached_towers', cache_name), 'rb') as handle:
                    block_set_towers = pickle.load(handle)
            except:
                return None, None
            
            # if it cached file exists, save it
            self.cached_towers = block_set_towers
            if (key in block_set_towers) and \
              (n_tower in block_set_towers[key]) and \
              (len(block_set_towers[key][n_tower][0]) >= self.n_samples[num_blocks]):
                print('Using random planning towers from cache for tower %d' % n_tower)
                self.using_cache = True
                return self.cached_towers[key][n_tower][:self.n_samples[num_blocks]]
            else:
                return None, None
        # if already loaded cache, get relevant towers
        else:
            try:
                towers, block_ids = self.cached_towers[key][n_tower][:self.n_samples[num_blocks]]
            except:
                return None, None
            print('Using saved random planning towers from cache for tower %d' % n_tower)
            return towers, block_ids

    # NOTE: this will not cache towers if some towers have already previously been found in the cache
    def cache_towers(self, args, towers, tower_block_ids, n_tower):
        if not self.using_cache:
            if args.block_set_fname == '':
                cache_name = 'random_set.pkl'
            else:
                cache_name = os.path.basename(args.block_set_fname)
                
            num_blocks = len(towers[0])
            # see if cache exists
            try:
                with open(os.path.join('learning/evaluate/cached_towers', cache_name), 'rb') as handle:
                    block_set_towers = pickle.load(handle)
            except:
                block_set_towers = {}
            if '%dblock'%num_blocks not in block_set_towers:
                block_set_towers['%dblock'%num_blocks] = {}    
            print('Saving randomly generated towers for tower %d' % n_tower)
            block_set_towers['%dblock'%num_blocks][n_tower] = (towers, tower_block_ids)
                
            with open(os.path.join('learning/evaluate/cached_towers', cache_name), 'wb') as handle:
                pickle.dump(block_set_towers, handle)    
        
    def generate_candidate_towers(self, blocks, args, num_blocks=None, n_tower=None):
        if num_blocks is None:
            num_blocks = len(blocks)
        #tower_vectors, tower_block_ids = self.get_cached_towers(args, num_blocks, blocks, n_tower)
        tower_vectors = None
        if tower_vectors is None:
            tower_vectors = []
            tower_block_ids = []
            for _ in range(0, self.n_samples[num_blocks]):
                tower, rotated_tower = sample_random_tower(blocks, num_blocks=num_blocks, \
                                            ret_rotated=True, discrete=False)
                tower_vectors.append([b.vectorize() for b in rotated_tower])
                tower_block_ids.append([b.get_id() for b in rotated_tower])
            #self.cache_towers(args, tower_vectors, tower_block_ids, n_tower)
        return tower_vectors, tower_block_ids

    def plan(self, blocks, ensemble, reward_fn, args, num_blocks=None, n_tower=None):
        #n = len(blocks)
        #max_height = 0
        #max_tower = []

        # Step (1): Build dataset of potential towers. 
        tower_vectors, tower_block_ids = self.generate_candidate_towers(blocks, args, num_blocks, n_tower)
        
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
            n_estimate = 10
            p_stables = np.zeros(len(tower_vectors))
            for ti, tower_vec in enumerate(tower_vectors):
                # estimate prob of constructability
                results = np.ones(n_estimate)
                all_stable = 1.
                for n in range(n_estimate):
                    noisy_tower = []
                    for block_vec in tower_vec:
                        noisy_block = deepcopy(block_vec)
                        noisy_block[7:9] += np.random.randn(2)*args.plan_xy_noise
                        noisy_tower.append(noisy_block)
                    block_tower = [Object.from_vector(block) for block in noisy_tower]
                    if not self.tp.tower_is_constructable(block_tower):
                        results[n] = 0.
                        all_stable = 0.
                        break
                p_stables[ti] = all_stable # np.mean(results)
                
        elif args.planning_model == 'simple-model':
            p_stables = np.zeros(len(tower_vectors))
            for ti, tower_vec in enumerate(tower_vectors):
                block_tower = [Object.from_vector(block) for block in tower_vec]
                if self.tp.tower_is_constructable(block_tower):
                    p_stables[ti] = 1.

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

class LatentEnsemblePlanner:
    def __init__(self, logger, n_samples=None, R_unstable=0.):
        self.tower_keys = ['2block', '3block', '4block', '5block']
        self.n_samples = {2: 5000, 3: 10000, 4: 20000,  5: 100000, 6: 250000, 7:500000}
        self.tp = TowerPlanner(stability_mode='contains')
        self.logger = logger
        self.using_cache = False
        self.R_unstable = R_unstable

    def get_cached_towers(self, args, num_blocks, blocks, n_tower):
        key = '%dblock' % num_blocks
        if not self.using_cache:
            if args.block_set_fname == '':
                cache_name = 'random_set.pkl'
            else:
                cache_name = os.path.basename(args.block_set_fname)
            
            # see if cache for this block set exists    
            try:
                with open(os.path.join('learning/evaluate/cached_towers', cache_name), 'rb') as handle:
                    block_set_towers = pickle.load(handle)
            except:
                return None, None
            
            # if it cached file exists, save it
            self.cached_towers = block_set_towers
            if (key in block_set_towers) and \
              (n_tower in block_set_towers[key]) and \
              (len(block_set_towers[key][n_tower][0]) >= self.n_samples[num_blocks]):
                print('Using random planning towers from cache for tower %d' % n_tower)
                self.using_cache = True
                return self.cached_towers[key][n_tower][:self.n_samples[num_blocks]]
            else:
                return None, None
        # if already loaded cache, get relevant towers
        else:
            try:
                towers, block_ids = self.cached_towers[key][n_tower][:self.n_samples[num_blocks]]
            except:
                return None, None
            print('Using saved random planning towers from cache for tower %d' % n_tower)
            return towers, block_ids

    # NOTE: this will not cache towers if some towers have already previously been found in the cache
    def cache_towers(self, args, towers, tower_block_ids, n_tower):
        if not self.using_cache:
            if args.block_set_fname == '':
                cache_name = 'random_set.pkl'
            else:
                cache_name = os.path.basename(args.block_set_fname)
                
            num_blocks = len(towers[0])
            # see if cache exists
            try:
                with open(os.path.join('learning/evaluate/cached_towers', cache_name), 'rb') as handle:
                    block_set_towers = pickle.load(handle)
            except:
                block_set_towers = {}
            if '%dblock'%num_blocks not in block_set_towers:
                block_set_towers['%dblock'%num_blocks] = {}    
            print('Saving randomly generated towers for tower %d' % n_tower)
            block_set_towers['%dblock'%num_blocks][n_tower] = (towers, tower_block_ids)
                
            with open(os.path.join('learning/evaluate/cached_towers', cache_name), 'wb') as handle:
                pickle.dump(block_set_towers, handle)    
        
    def generate_candidate_towers(self, blocks, args, num_blocks=None, n_tower=None, fixed_order=False):
        num_blocks = len(blocks)
        
        tower_vectors = []
        rot_tower_vectors = []
        tower_block_ids = []
        for _ in range(0, self.n_samples[num_blocks]):
            tower = sample_random_tower(blocks, num_blocks=num_blocks, \
                                        ret_rotated=False, discrete=False, fixed_order=fixed_order)
            rotated_tower = [get_rotated_block(b) for b in tower]

            tower_vectors.append([b.vectorize() for b in tower])
            tower_block_ids.append([b.get_id() for b in tower])
            rot_tower_vectors.append([b.vectorize() for b in rotated_tower])
        return tower_vectors, tower_block_ids, rot_tower_vectors

    def plan(self, blocks, ensemble, reward_fn, args, num_blocks=None, n_tower=None, latent_samples=None, pf_latent_ix=-1, fixed_order=False):
        # Step (1): Build dataset of potential towers. 
        tower_vectors, tower_block_ids, rotated_tower_vectors = self.generate_candidate_towers(blocks, args, num_blocks, n_tower, fixed_order=fixed_order)
        
        # Step (2): Get predictions for each tower.
        towers = np.array(tower_vectors)
        block_ids = np.array(tower_block_ids)
        rotated_towers = np.array(rotated_tower_vectors)

        # Since we are only planning for towers of a single size,
        # always use the '2block' key for simplicity. The rest currently
        # need at least some data for the code to work.
        valid_ixs = np.arange(0, towers.shape[0])
        
        n_blocks = towers.shape[1]
        sub_tower_preds = torch.zeros(towers.shape[0], n_blocks-1)
        for n_blocks in range(2, n_blocks+1):
            n_towers = len(valid_ixs)
            # Create dataset based on current valid_ixs.
            labels = np.zeros((valid_ixs.shape[0],))
            tower_dict = {}
            for k in self.tower_keys:
                tower_dict[k] = {}
                if k == '2block':
                    tower_dict[k]['towers'] = towers[valid_ixs, :n_blocks, :]
                    tower_dict[k]['labels'] = labels
                    tower_dict[k]['block_ids'] = block_ids[valid_ixs, :n_blocks]
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
            sub_tower_pred = []
            for tensor, b_ids, _ in tower_loader:
                # sub_tower_preds = []
                b_ids = b_ids.long()
                if torch.cuda.is_available():
                    tensor = tensor.cuda()
                    b_ids = b_ids.cuda()
                with torch.no_grad():
                    if pf_latent_ix > -1:
                        pred = ensemble.forward(tensor[:, :, 4:], 
                                                b_ids,
                                                N_samples=50,
                                                collapse_ensemble=True,
                                                collapse_latents=True,
                                                pf_latent_ix=pf_latent_ix,
                                                latent_samples=latent_samples).squeeze(-1)
                    else:
                        pred = ensemble.forward(tensor[:, :, 4:], 
                                                b_ids,
                                                N_samples=10,
                                                collapse_ensemble=True,
                                                collapse_latents=True).squeeze(-1)
                    sub_tower_pred.append(pred)

            sub_tower_pred = torch.cat(sub_tower_pred, dim=0)[:n_towers].cpu()
            sub_tower_preds[valid_ixs, n_blocks-2] = sub_tower_pred[:,0]
            # sub_tower_preds[valid_ixs, n_blocks-2] = sub_tower_preds.prod(dim=0)

            # Remove towers that will not confidently result in stable towers.
            likely_stable = sub_tower_pred[:, 0] > 0.3
            valid_ixs = valid_ixs[likely_stable]
            print('Remaining: %d' % len(valid_ixs))

        p_stables = sub_tower_preds.prod(dim=1)
        # print('INVALID:', (p_stables==0).sum())

        # Step (3): Find the tallest tower of a given height.
        max_reward, max_exp_reward, max_tower, max_stable, max_rotated = -100, -100, None, 0, None
        ground_truth = -100
        max_reward_block_ids = None 
        #print(p_stables.shape, towers.shape, block_ids.shape, rotated_towers.shape)
        for ix, (p, tower, tower_block_ids, rotated_tower) in enumerate(zip(p_stables, towers, block_ids, rotated_towers)):
            # print(rotated_tower)
            # print(tower)
            # print(p)
            # print('-----')
            reward = reward_fn(rotated_tower)
            exp_reward = (1-p)*self.R_unstable + p*reward
            if exp_reward >= max_exp_reward:# and p > 0.8:
                #print(p, reward, exp_reward)
                if exp_reward > max_exp_reward or (exp_reward == max_exp_reward and p > max_stable):
                    max_tower = tower_vectors[ix]
                    max_reward = reward
                    max_exp_reward = exp_reward
                    max_stable = p
                    max_reward_block_ids = tower_block_ids
                    max_rotated = rotated_tower_vectors[ix]

            # Check ground truth stability to find maximum reward.
            if self.tp.tower_is_constructable([Object.from_vector(rotated_tower[bx,:]) for bx in range(tower.shape[0])]) \
                and reward > ground_truth:
                ground_truth = reward

        if max_tower is None:
            print('None Found')
            max_tower = tower_vectors[0]
            max_reward = reward_fn(towers[0])
            max_rotated = rotated_towers[0]
        



        print('Prob Stable:', max_stable, max_reward)

        return max_tower, max_reward, ground_truth, max_reward_block_ids, max_rotated
