from copy import copy
import random
import csv
import numpy as np

from tamp.predicates import On

class Object:
    def __init__(self, num):
        self.num = num
        
class Block(Object):
    def __init__(self, num):
        super(Block, self).__init__(num)
        self.name = 'block_%i' % num

class Table(Object):
    def __init__(self):
        super(Table, self).__init__(0) # make 1 when using * as 0
        self.name = 'table'

'''
class Star:
    def __init__(self):
        self.name= 'star'
        self.num = 0
'''
class ABCBlocksWorld:
    def __init__(self, num_blocks):
        '''
        remove for now 0: * object
        features:
            0: Table
            1 --> num_blocks-1: Blocks
        '''
        self.num_blocks = num_blocks
        self.num_objects = num_blocks + 1 # table is also an object
        self.min_block = 1                # 0 is the table
        self.max_block = num_blocks
        
        self._blocks = {i: Block(i) for i in range(self.min_block, self.max_block+1)}
        self._table = Table()
        #self._star = Star()
        self._stacked_blocks = None
        self.reset()
        
    def reset(self):
        self._stacked_blocks = []

    def transition(self, action):
        if action is not None:
            bottom_block_num = action[0].num
            top_block_num = action[1].num
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
        action = None
        remaining_blocks = list(set(self._blocks.values()).difference(set(self._stacked_blocks)))
        if len(remaining_blocks) > 0:
            top_block = random.choice(remaining_blocks)
            if len(self._stacked_blocks) > 0:
                bottom_block = self._stacked_blocks[-1]
                action = (bottom_block, top_block)
            else:
                possible_bottom_blocks = list(set(remaining_blocks).difference(set([top_block])))
                if len(possible_bottom_blocks) > 0:
                    bottom_block = random.choice(possible_bottom_blocks)
                    action = (bottom_block, top_block)
        return action
            
    def expert_policy(self):
        action = None
        if len(self._stacked_blocks) > 0:
            bottom_block = self._stacked_blocks[-1]
            if bottom_block.num != self.max_block:
                top_block = self.get_object_by_num(bottom_block.num + 1)
                action = (bottom_block, top_block)
        else:
            random_bottom_block = random.choice(list(self._blocks.values()))
            random_top_block = random.choice(list(self._blocks.values()))
            action = (random_bottom_block, random_top_block)
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

    def get_obj_one_hot(self, obj_i):
        one_hot = np.zeros(self.num_objects)
        one_hot[obj_i] = 1.
        return one_hot

    def get_obj_num(self, one_hot):
        return int(np.where(one_hot == 1.)[0])

    def get_object_by_num(self, num):
        for block in self._blocks.values():
            if block.num == num:
                return block

    def get_vectorized_state(self):
        def get_int_object(object):
            if isinstance(object, Table):
                return self._table.num
            elif isinstance(object, Block):
                return object.num
            elif object == '*':
                return STAR
            
        #object_features = np.eye(MAX_OBJECTS)
        object_features = np.expand_dims(np.arange(self.num_objects), 1)
        edge_features = np.zeros((self.num_objects, self.num_objects, 1))

        # edge_feature[i, j, 0] == 1 if j on i, else 0
        state = self.get_state()
        for fluent in state:
            bottom_i = get_int_object(fluent.bottom)
            top_i = get_int_object(fluent.top)
            edge_features[bottom_i, top_i, 0] = 1.
            
        return object_features, edge_features

    def get_vectorized_action(self, action):
        #action_vec = np.zeros(2*MAX_OBJECTS)
        action_vec = np.zeros(2)
        if action is not None:
            bottom_block, top_block = action
            action_vec[0] = bottom_block.num
            action_vec[1] = top_block.num
            #action_vec[:MAX_OBJECTS] = self.get_obj_one_hot(bottom_block.num)
            #action_vec[MAX_OBJECTS:] = self.get_obj_one_hot(top_block.num)
        return action_vec