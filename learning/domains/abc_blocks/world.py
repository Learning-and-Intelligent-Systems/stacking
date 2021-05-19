from copy import copy
import random
import csv
import numpy as np

from tamp.predicates import On

'''
features:
    0: * object
    1: Table
    2-6: Blocks and Place block action
    7: non action
    
'''
STAR=0
TABLE=1
MINBLOCK=2
MAXBLOCK=6
NONACTION=7

def get_vectorized_state(state):
    def get_int_object(object):
        if isinstance(object, Table):
            return TABLE
        elif isinstance(object, Block):
            return object.num
        elif object == '*':
            return STAR
    edges = np.zeros((7,7))
    for fluent in state:
        bottom_i = get_int_object(fluent.bottom)
        top_i = get_int_object(fluent.top)
        edges[bottom_i, top_i] = 1
    return edges
    
class Block:
    def __init__(self, num):
        self.name = 'block_%i' % num
        self.num = num

class Table:
    def __init__(self):
        self.name = 'table'

class Star:
    def __init__(self):
        self.name= 'star'

class ABCBlocksWorld:
    def __init__(self):
        self._blocks = {i: Block(i) for i in range(MINBLOCK, MAXBLOCK+1)}
        self._table = Table()
        self._star = Star()
        self._stacked_blocks = None
        self.reset()
        
    def reset(self):
        self._stacked_blocks = []

    def transition(self, action):
        if action != NONACTION:
            if len(self._stacked_blocks) == 0:
                self._stacked_blocks.append(self._blocks[action])
            else:
                if action == self._stacked_blocks[-1].num+1:
                    self._stacked_blocks.append(self._blocks[action])
        
    def random_policy(self):
        remaining_blocks = list(set(self._blocks.values()).difference(set(self._stacked_blocks)))
        if len(remaining_blocks) > 0:
            action = random.choice(remaining_blocks).num
            return action
        else:
            return NONACTION
            
    def expert_policy(self):
        if len(self._stacked_blocks) > 0:
            top = self._stacked_blocks[-1]
            if top.num != MAXBLOCK:
                return top.num+1
            else:
                return NONACTION
        else:
            random_block = random.choice(list(self._blocks.values()))
            return random_block.num
            
    def get_state(self):
        state = []
        # bottom stacked block is on table
        if len(self._stacked_blocks) > 0:
            state.append(On(self._table, self._stacked_blocks[-1]))
        # remaining stacked blocks
        for bottom_block, top_block in zip(self._stacked_blocks[:-1], self._stacked_blocks[1:]):
            state.append(On(bottom_block, top_block))
        # remaining blocks on table
        for block in self._blocks.values():
            if block not in self._stacked_blocks:
                state.append(On(self._table, block))
        return state
        
    def parse_goals_csv(self, goal_file_path):
        def ground_obj(obj_str):
            if obj_str == 'table':
                return self._table
            elif obj_str == '*':
                return self._star
            else:
                return self._blocks[int(obj_str)]
        goals = []
        with open(goal_file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                pred_name = row[0]
                pred_args = row[1:]
                if pred_name == 'On':
                    goals.append([On(ground_obj(pred_args[0]), ground_obj(pred_args[1]))])
        return goals
                    