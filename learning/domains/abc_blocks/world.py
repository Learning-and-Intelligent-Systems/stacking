from copy import copy
import random
import csv
import numpy as np

from tamp.predicates import On

'''
remove for now 0: * object
features:
    0: Table
    1 --> N_OBJECTS-1: Blocks and Place block action
'''
N_OBJECTS=3 # must be 2 or greater
#STAR=0
TABLE=0
MINBLOCK=1
MAXBLOCK=N_OBJECTS-1

# NOTE: all object properties are currently static
#       one-hot encoding of the index of the object
object_features = np.eye(N_OBJECTS)

def get_obj_one_hot(obj_i):
    one_hot = np.zeros(N_OBJECTS)
    one_hot[obj_i] = 1.
    return one_hot

def get_obj_num(one_hot):
    return int(np.where(one_hot == 1.)[0])
    
def get_vectorized_state(state):
    def get_int_object(object):
        if isinstance(object, Table):
            return TABLE
        elif isinstance(object, Block):
            return object.num
        elif object == '*':
            return STAR
    edge_features = np.zeros((N_OBJECTS, N_OBJECTS, 2*N_OBJECTS+1))
    
    # edge_feature[i, j, 0] == 1 if j on i, else 0
    for fluent in state:
        bottom_i = get_int_object(fluent.bottom)
        top_i = get_int_object(fluent.top)
        edge_features[bottom_i, top_i, 0] = 1.
        
    # edge_features[i,j,1:] are the 1-hot encodings of the objects
    for bottom_obj_i in range(N_OBJECTS):
        for top_obj_i in range(N_OBJECTS):
            edge_features[bottom_obj_i, top_obj_i, 1:N_OBJECTS+1] = get_obj_one_hot(bottom_obj_i)
            edge_features[bottom_obj_i, top_obj_i, 1+N_OBJECTS:] = get_obj_one_hot(top_obj_i)
    return edge_features
    
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
            bottom_block_num = get_obj_num(action[:N_OBJECTS])
            top_block_num = get_obj_num(action[N_OBJECTS:])
            # can only stack blocks by increments of one
            if top_block_num == bottom_block_num + 1:
                # if this is the start of the stack add both blocks to stacked list
                if len(self._stacked_blocks) == 0:
                    self._stacked_blocks.append(self._blocks[bottom_block_num])
                    self._stacked_blocks.append(self._blocks[top_block_num])
                # can only build one stack at a time (for now)
                else:
                    self._stacked_blocks.append(self._blocks[top_block_num])
        
    def random_policy(self):
        action = np.zeros(2*N_OBJECTS)
        remaining_blocks = list(set(self._blocks.values()).difference(set(self._stacked_blocks)))
        if len(remaining_blocks) > 0:
            top_block = random.choice(remaining_blocks)
            if len(self._stacked_blocks) > 0:
                bottom_block = self._stacked_blocks[-1]
                action[:N_OBJECTS] = get_obj_one_hot(bottom_block.num)
                action[N_OBJECTS:] = get_obj_one_hot(top_block.num)
            else:
                possible_bottom_blocks = list(set(remaining_blocks).difference(set([top_block])))
                if len(possible_bottom_blocks) > 0:
                    bottom_block = random.choice(possible_bottom_blocks)
                    action[:N_OBJECTS] = get_obj_one_hot(bottom_block.num)
                    action[N_OBJECTS:] = get_obj_one_hot(top_block.num)
        return action
            
    def expert_policy(self):
        action = np.zeros(2*N_OBJECTS)
        if len(self._stacked_blocks) > 0:
            top_of_stack = self._stacked_blocks[-1]
            if top_of_stack.num != MAXBLOCK:
                action[:N_OBJECTS] = get_obj_one_hot(top_of_stack.num)
                action[N_OBJECTS:] = get_obj_one_hot(top_of_stack.num+1)
        else:
            random_bottom_block = random.choice(list(self._blocks.values()))
            random_top_block = random.choice(list(self._blocks.values()))
            action[:N_OBJECTS] = get_obj_one_hot(random_bottom_block.num)
            action[N_OBJECTS:] = get_obj_one_hot(random_top_block.num)
        return action
            
    def get_state(self):
        state = []
        # bottom stacked block is on table
        if len(self._stacked_blocks) > 0:
            state.append(On(self._table, self._stacked_blocks[0]))
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
                    