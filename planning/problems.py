from copy import copy
from collections import namedtuple
import numpy as np

import torch

from block_utils import ZERO_POS, Pose
from planning.utils import make_tower_dataset, random_placement
from tower_planner import TowerPlanner

class Problem:
    def __init__(self):
        pass
        
    def sample_actions(self, node_value, model):
        pass
        
    def reward_fn(self, towers, model):
        pass
        
class Tallest(Problem):
    def __init__(self, max_height):
        self.samples_per_block = 5
        self.max_height = max_height
        self.tp = TowerPlanner(stability_mode='contains')
        
    def sample_actions(self, parent_node, model, discrete=False):
        new_towers = []
        new_blocks_remaining = []
        if len(parent_node['tower']) == 0:
            # first action: place each possible block at (0,0) at a random orientation
            for block in parent_node['blocks_remaining']:
                for _ in range(self.samples_per_block):
                    tower = random_placement(block, [], discrete=discrete)
                    tower[0].pose = Pose(ZERO_POS, tower[0].pose.orn)
                    blocks_remaining = copy(parent_node['blocks_remaining'])
                    blocks_remaining.remove(block)
                    new_towers.append(tower)
                    new_blocks_remaining.append(blocks_remaining)
        else:
            # randomly sample a placement of a random block
            for block in parent_node['blocks_remaining']:
                blocks_remaining = copy(parent_node['blocks_remaining'])
                blocks_remaining.remove(block)
                for _ in range(self.samples_per_block):
                    tower = random_placement(block, parent_node['tower'], discrete=discrete)
                    new_towers.append(tower)
                    new_blocks_remaining.append(blocks_remaining)
                    
        all_rewards = self.reward_fn(new_towers, model)

        terms = [False]*len(new_towers)
        for i, tower in enumerate(new_towers):
            # rewards of 0 are unstable --> terminal nodes
            # once max height is reached --> terminal nodes
            if all_rewards['exp_reward'][i] == 0 or len(tower) == self.max_height:
                terms[i] = True
                
        new_nodes = []
        for i, (tower, blocks_remaining, term) in enumerate(zip(new_towers, new_blocks_remaining, terms)):
            new_node =  {'parent': None, 
                                'children': [],
                                'term': term,
                                'leaf': True,
                                'exp_reward': all_rewards['exp_reward'][i], 
                                'value': 0,
                                'count': 0,
                                'tower': tower,
                                'blocks_remaining': blocks_remaining,
                                'tower_height': all_rewards['reward'][i],
                                'ground_truth': all_rewards['ground_truth'][i]}
            new_nodes.append(new_node)
        return new_nodes
    '''
    def sample_action(self, parent_node, model, discrete=False):
        block = np.random.choice(parent_node['blocks_remaining'])
        if len(parent_node['tower']) == 0:
            # first action: place block at (0,0)
            block.pose = Pose(ZERO_POS, block.pose.orn)
            tower = [block]
        else:
            # randomly sample a placement of a random block
            tower = random_placement(block, parent_node['tower'], discrete=discrete)
        blocks_remaining = copy(parent_node['blocks_remaining'])
        blocks_remaining.remove(block)
        
        all_rewards = self.reward_fn([tower], model)
        
        # rewards of 0 are unstable --> terminal nodes
        # once max height is reached --> terminal nodes
        term = False
        if all_rewards['exp_reward'][0] == 0 or len(tower) == self.max_height:
            term = True
            
        new_node =  {'parent': None, 
                        'children': [],
                        'term': term,
                        'leaf': True,
                        'exp_reward': all_rewards['exp_reward'][0], 
                        'value': 0,
                        'count': 0,
                        'tower': tower,
                        'blocks_remaining': blocks_remaining,
                        'tower_height': all_rewards['reward'][0],
                        'ground_truth': all_rewards['ground_truth'][0]}
    
        return new_node
    '''
    def reward_fn(self, towers, model):
        all_rewards = {'exp_reward': [], 'reward': [], 'ground_truth': []}
        if len(towers[0]) == 1:
            # only single block towers, always stable
            reward = [tower[0].dimensions[2] for tower in towers]
            all_rewards['exp_reward'] = reward
            all_rewards['reward'] = reward
            all_rewards['ground_truth'] = reward
        else:
            tower_loader = make_tower_dataset(towers)
            # this assumes there is only one batch
            for tensor, _ in tower_loader:
                with torch.no_grad():
                    preds = model.forward(tensor)
            p_stables = preds.mean(dim=1) # average ensemble output
            
            exp_rewards = []
            rewards = []
            ground_truths = []
            for ix, (p, tower) in enumerate(zip(p_stables, towers)):
                tower_height = np.sum([block.dimensions[2] for block in tower])
                rewards += [tower_height]
                if p > 0.5: # stable
                    exp_rewards += [float(p*tower_height)]
                else:
                    exp_rewards += [0]
                ground_truths += [self.tp.tower_is_constructable(tower)*tower_height]
            all_rewards['exp_reward'] = exp_rewards
            all_rewards['reward'] = rewards
            all_rewards['ground_truth'] = ground_truths
        return all_rewards

class Overhang(Problem):
    def __init__(self):
        pass
        
    def sample_actions(self, node_value, model):
        pass
        
    def reward_fn(self, towers, model):
        pass
            
class Deconstruct(Problem):
    def __init__(self):
        pass
        
    def sample_actions(self, node_value, model):
        pass
        
    def reward_fn(self, towers, model):
        pass