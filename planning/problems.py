from copy import copy
from collections import namedtuple
import numpy as np

import torch

from block_utils import ZERO_POS, Pose
from planning.utils import make_tower_dataset, random_placement
from planning.tree import NodeValue
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
        
    def sample_actions(self, node_value, model):
        new_values = []
        eval_towers = []
        if len(node_value.tower) == 0:
            # first action: place each possible block at (0,0)
            for block in node_value.blocks_remaining:
                block.pose = Pose(ZERO_POS, block.pose.orn)
                new_blocks_remaining = copy(node_value.blocks_remaining)
                new_blocks_remaining.remove(block)
                new_values.append(NodeValue([block], new_blocks_remaining))
                eval_towers.append([block])
        else:
            # randomly sample a placement of a random block
            for block in node_value.blocks_remaining:
                new_blocks_remaining = copy(node_value.blocks_remaining)
                new_blocks_remaining.remove(block)
                for _ in range(self.samples_per_block):
                    new_tower = random_placement(block, node_value.tower)
                    new_values.append(NodeValue(new_tower, new_blocks_remaining))
                    eval_towers.append(new_tower)
                    
        all_rewards = self.reward_fn(eval_towers, model)

        terms = [False]*len(eval_towers)
        for i, tower in enumerate(eval_towers):
            # rewards of 0 are unstable --> terminal nodes
            # once max height is reached --> terminal nodes
            if all_rewards['exp_reward'][i] == 0 or len(tower) == self.max_height:
                terms[i] = [True]
        return new_values, all_rewards, terms

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