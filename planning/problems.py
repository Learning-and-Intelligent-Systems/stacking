from copy import copy
from collections import namedtuple
import numpy as np

import torch

from block_utils import ZERO_POS, Pose
from planning.utils import make_tower_dataset, random_placement
from planning.tree import NodeValue

class Problem:
    def __init__(self):
        pass
        
    def sample_actions(self, node_value, model):
        pass
        
    def cost_fn(self, towers, model):
        pass
        
class Tallest(Problem):
    def __init__(self):
        self.samples_per_block = 5
        self.max_height = 5
        
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
                    
        costs = self.cost_fn(eval_towers, model)
        if len(eval_towers[0]) == self.max_height:
            leaves = [True]*len(eval_towers)
        else:
            leaves = [False]*len(eval_towers)
        return new_values, costs, leaves

    def cost_fn(self, towers, model):
        if len(towers[0]) == 1:
            # only single block towers, always stable, but use 0 so all 
            # equally likely to get selected
            costs = np.zeros((len(towers)))
        else:
            tower_loader = make_tower_dataset(towers)
            # this assumes there is only one batch
            for tensor, _ in tower_loader:
                with torch.no_grad():
                    preds = model.forward(tensor)
            p_stables = preds.mean(dim=1) # average ensemble output
            costs = []
            for ix, (p, tower) in enumerate(zip(p_stables, towers)):
                if p > 0.5: # TODO: stable?
                    tower_height = np.sum([block.dimensions[2] for block in tower])
                    costs += [float(p*tower_height)]
                else:
                    costs += [0]
        return costs

class Overhang(Problem):
    def __init__(self):
        pass
        
    def sample_actions(self, node_value, model):
        pass
        
    def cost_fn(self, towers, model):
        pass
            
class Deconstruct(Problem):
    def __init__(self):
        pass
        
    def sample_actions(self, node_value, model):
        pass
        
    def cost_fn(self, towers, model):
        pass