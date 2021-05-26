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
'''
N_OBJECTS=7 # must be 3 or greater
STAR=0
TABLE=1
MINBLOCK=2
MAXBLOCK=N_OBJECTS-1

# NOTE: all object properties are currently static
#       one-hot encoding of the index of the object
object_features = np.eye(N_OBJECTS)

def get_vectorized_state(state):
    def get_int_object(object):
        if isinstance(object, Table):
            return TABLE
        elif isinstance(object, Block):
            return object.num
        elif object == '*':
            return STAR
    edge_features = np.zeros((N_OBJECTS, N_OBJECTS, 1))
    for fluent in state:
        bottom_i = get_int_object(fluent.bottom)
        top_i = get_int_object(fluent.top)
        edge_features[bottom_i, top_i, :] = 1.
    return object_features, edge_features
    
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
        if action.sum() != 0.:
            block_num = int(np.where(action == 1.)[0])
            if len(self._stacked_blocks) == 0 or block_num == self._stacked_blocks[-1].num+1:
                self._stacked_blocks.append(self._blocks[block_num])
        
    def random_policy(self):
        action = np.zeros(MAXBLOCK+1)
        remaining_blocks = list(set(self._blocks.values()).difference(set(self._stacked_blocks)))
        if len(remaining_blocks) > 0:
            block_num = random.choice(remaining_blocks).num
            action[block_num] = 1.
        return action
            
    def expert_policy(self):
        action = np.zeros(MAXBLOCK+1)
        if len(self._stacked_blocks) > 0:
            top = self._stacked_blocks[-1]
            if top.num != MAXBLOCK:
                action[top.num+1] = 1.
        else:
            random_block = random.choice(list(self._blocks.values()))
            action[random_block.num] = 1.
        return action
            
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
                    