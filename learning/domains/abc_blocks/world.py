from copy import copy
import csv
import numpy as np

from tamp.predicates import On

### Object Classes

class Object:
    def __init__(self, num):
        self.num = num


class Block(Object):
    def __init__(self, num):
        super(Block, self).__init__(num)
        self.name = 'block_%i' % num


class Table(Object):
    def __init__(self, num):
        super(Table, self).__init__(num) # make 1 when using * as 0
        self.name = 'table'


'''
class Star:
    def __init__(self):
        self.name= 'star'
        self.num = 0
'''


### World and State Classes

class LogicalState:
    def __init__(self, blocks, num_objects):
        self.table_num = 0
        self.table = Table(self.table_num)
        self.blocks = blocks
        self.stacked_blocks = []
        self.num_objects = num_objects

    def as_logical(self):
        logical_state = []

        # stacked blocks
        if len(self.stacked_blocks) > 0:
            logical_state.append(On(self.table, self.stacked_blocks[0]))
        for bottom_block, top_block in zip(self.stacked_blocks[:-1], self.stacked_blocks[1:]):
            logical_state.append(On(bottom_block, top_block))

        # remaining blocks on table
        for block in self.blocks.values():
            if block not in self.stacked_blocks:
                logical_state.append(On(self.table, block))

        return logical_state

    def as_vec(self):
        object_features = np.expand_dims(np.arange(self.num_objects), 1)
        edge_features = np.zeros((self.num_objects, self.num_objects, 1))

        # edge_feature[i, j, 0] == 1 if j on i, else 0
        for predicate in self.as_logical():
            bottom_i = predicate.bottom.num
            top_i = predicate.top.num
            edge_features[bottom_i, top_i, 0] = 1.

        return object_features, edge_features


# Ground Truth Blocks World
class ABCBlocksWorldGT:
    def __init__(self, args, num_blocks):
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

    def get_init_state(self):
        return LogicalState(self._blocks, self.num_objects)

    def transition(self, state, action):
        new_state = copy(state)
        if action is not None:
            bottom_block_num = action[0]
            top_block_num = action[1]
            # can only stack blocks by increments of one
            if top_block_num == bottom_block_num + 1:
                # if this is the start of the stack add both blocks to stacked list
                if len(state.stacked_blocks) == 0:
                    new_state.stacked_blocks.append(self._blocks[bottom_block_num])
                    new_state.stacked_blocks.append(self._blocks[top_block_num])
                # can only build one stack at a time (for now)
                else:
                    new_state.stacked_blocks.append(self._blocks[top_block_num])
        return new_state

    def random_policy(self, state):
        action = None
        top_block_num = np.random.choice(list(self._blocks))
        bottom_block_num = np.random.choice(list(self._blocks))
        return (bottom_block_num, top_block_num)

    # attempt to stack a block that is currently on the table
    def random_remaining_policy(self, state):
        action = None
        remaining_blocks = list(set(self._blocks.values()).difference(set(self.stacked_blocks)))
        if len(remaining_blocks) > 0:
            top_block_idx = np.random.choice(len(remaining_blocks))
            top_block_num = remaining_blocks[top_block_idx].num
            if len(state.stacked_blocks) > 0:
                bottom_block_num = state.stacked_blocks[-1].num
                action = (bottom_block_num, top_block_num)
            else:
                possible_bottom_blocks = list(set(remaining_blocks).difference(set([top_block])))
                if len(possible_bottom_blocks) > 0:
                    bottom_block_idx = np.random.choice(len(possible_bottom_blocks))
                    bottom_block_num = possible_bottom_blocks[bottom_block_idx].num
                    action = (bottom_block_num, top_block_num)
        return action

    def expert_policy(self, state):
        action = None
        if len(state.stacked_blocks) > 0:
            bottom_block_num = state.stacked_blocks[-1].num
            if bottom_block_num != self.max_block:
                top_block_num = bottom_block_num + 1
                action = (bottom_block_num, top_block_num)
        else:
            random_bottom_block_num = np.random.choice(list(self._blocks.values()))
            random_top_block_num = np.random.choice(list(self._blocks.values()))
            action = (random_bottom_block_num, random_top_block_num)
        return action


# When using learned model for transitions, edge states won't always make sense as logical states,
# so need a separate class (eg. 2 blocks can be on top of one in a vectorized edge state)
class VectorizedState:
    def __init__(self, blocks, num_objects):
        self.table_num = 0
        self.table = Table(self.table_num)
        self.object_features = np.expand_dims(np.arange(num_objects), 1)
        self.edge_features = np.zeros((num_objects, num_objects, 1))

        # edge_feature[i, j, 0] == 1 if j on i, else 0
        # initially everything on table
        for block_num in blocks.keys():
            self.edge_features[self.table.num, block_num, 0] = 1.


# Learned Blocks World
class ABCBlocksWorldLearned:
    def __init__(self, args, num_blocks):
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

        if args.model_path:
            self.model_path = args.model_path

    def get_init_state(self):
        return VectorizedState(self._blocks, self.num_objects)

    def transition(self, state, action):
        new_state = copy(state)
        vec_action = get_vectorized_action(action)
        model = torch.load_state(self.model_path)
        delta_edge_features = model(state.object_features, state.edge_features, vec_action)
        new_state.edge_features += delta_edge_features # for now object features are static
        return new_state


### Helper Functions
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
