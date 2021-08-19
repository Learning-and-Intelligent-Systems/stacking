from copy import copy, deepcopy
import csv
import numpy as np

from tamp.predicates import On
from learning.domains.abc_blocks.abc_blocks_data import model_forward

TABLE_NUM = 0

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


### World and State Classes

class ABCBlocksWorld:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.num_objects = num_blocks + 1 # table is also an object
        self.min_block_num = 1                # 0 is the table
        self.max_block_num = num_blocks

        self.table = Table(TABLE_NUM)
        self._blocks = {i: Block(i) for i in range(self.min_block_num, self.max_block_num+1)}

    def get_init_state(self):
        raise NotImplementedError

    def transition(self, state, action):
        raise NotImplementedError

    def random_policy(self, state):
        action = None
        remaining_blocks = list(set(self._blocks.keys()).difference(set(state.stacked_blocks)))
        if len(remaining_blocks) > 0:
            top_block_idx = np.random.choice(len(remaining_blocks))
            top_block_num = remaining_blocks[top_block_idx]
            if len(state.stacked_blocks) > 0:
                bottom_block_num = state.stacked_blocks[-1]
                action = (bottom_block_num, top_block_num)
            else:
                possible_bottom_blocks = list(set(remaining_blocks).difference(set([top_block_num])))
                if len(possible_bottom_blocks) > 0:
                    bottom_block_idx = np.random.choice(len(possible_bottom_blocks))
                    bottom_block_num = possible_bottom_blocks[bottom_block_idx]
                    action = (bottom_block_num, top_block_num)
        return action

    def reward(self, state, goal):
        return 1 if self.is_goal_state(state, goal) else 0

    def is_goal_state(self, state, goal):
        raise NotImplementedError


# Ground Truth Blocks World (with option to behave optimistically)
class ABCBlocksWorldGT(ABCBlocksWorld):
    def __init__(self, num_blocks):
        super().__init__(num_blocks)

    def get_init_state(self):
        return LogicalState(self._blocks, self.num_objects, self.table)

    def transition(self, state, action, optimistic=False):
        new_state = state.copy()
        if action is not None:
            bottom_block_num = action[0]
            top_block_num = action[1]
            condition = top_block_num not in state.stacked_blocks # TODO: might be redundant after random policy
            if not optimistic:
                condition = condition and \
                        top_block_num == bottom_block_num + 1
            if condition:
                # add both if bottom block is on table and this is the start of the stack
                # TODO: I think this can just be if len(stack) == 0
                if bottom_block_num not in state.stacked_blocks and \
                                        len(state.stacked_blocks) == 0:
                    new_state.stacked_blocks.append(bottom_block_num)
                    new_state.stacked_blocks.append(top_block_num)
                # if bottom block is on top of stack, can stack top block (can only build one stack at a time for now)
                elif bottom_block_num == state.stacked_blocks[-1]:
                    new_state.stacked_blocks.append(top_block_num)
        #print('new stack:', new_state.stacked_blocks)
        return new_state

    def execute_plan(self, plan):
        trajectory = []
        state = self.get_init_state()
        if plan:
            for action in plan:
                trajectory.append([state, action])
                state = self.transition(state, action)
            trajectory.append([state, np.zeros(2)])
        return trajectory

    # expert once stack is started
    def expert_policy(self, state):
        action = None
        if len(state.stacked_blocks) > 0:
            bottom_block_num = state.stacked_blocks[-1]
            if bottom_block_num != self.max_block_num:
                top_block_num = bottom_block_num + 1
                action = (bottom_block_num, top_block_num)
        else: # start stack
            bottom_block_num = np.random.choice(self.max_block_num-1)+1
            top_block_num = bottom_block_num +1
            action = (bottom_block_num, top_block_num)
        return action

    # goal is list of predicates (returned by state.as_logical())
    def is_goal_state(self, state, goal):
        in_goal = True
        for goal_pred in goal:
            in_goal = in_goal and goal_pred.in_state(state.as_logical())
        return in_goal

    # NOTE: we only use this function to calculate the ground state steps to goal
    # given a LEGAL goal state and start state (eg. won't get goal of (3 on 1))
    def steps_to_goal(self, state, goal_state, optimistic=False):
        goal_tower_height = len(goal_state.stacked_blocks)
        # stack hasn't started yet (just need to create stack)
        if state.stacked_blocks == []:
            return goal_tower_height-1

        # already reached goal
        if self.is_goal_state(state, goal_state.as_logical()):
            return 0

        goal_bottom_block = goal_state.stacked_blocks[0]
        goal_top_block = goal_state.stacked_blocks[-1]

        # cannot get goal blocks under stack
        if goal_top_block <= state.stacked_blocks[0]:
            return 1000

        # stack is partially complete, just need to finish
        if goal_bottom_block in state.stacked_blocks:
            return goal_top_block - state.stacked_blocks[-1]

        # can get blocks on top of stack
        if goal_bottom_block > state.stacked_blocks[-1]:
            if optimistic:
                return goal_tower_height
            else:
                return goal_top_block - state.stacked_blocks[-1]

        print('unhandled case for calculating steps to goal')
        import pdb; pdb.set_trace()


# Learned Blocks World where learned model is a classifier on top of the ground truth optimistic model
class ABCBlocksWorldLearnedClass(ABCBlocksWorldGT):
    def __init__(self, num_blocks, model):
        super().__init__(num_blocks)
        self.model = model

    def transition(self, state, action):
        object_features, edge_features = state.as_vec()
        pred = model_forward(self.model, [object_features, edge_features, action]).squeeze(0)
        if pred.round() == 1:
            return super().transition(state, action, optimistic=True)
        else:
            return state

# When using learned model for transitions, edge states won't always make sense as logical states,
# so need a separate World class (eg. 2 blocks can be on top of one in a vectorized edge state)

# Learned Blocks World
class ABCBlocksWorldLearned(ABCBlocksWorld):
    def __init__(self, num_blocks, model):
        super().__init__(num_blocks)
        self.object_features = np.expand_dims(np.arange(self.num_objects), 1) # for now object features are static
        self.model = model

    def get_init_state(self):
        edge_features = np.zeros((self.num_objects, self.num_objects, 1))
        # edge_feature[i, j, 0] == 1 if j on i, else 0
        # initially everything on table
        for block_num in self._blocks.keys():
            edge_features[self.table.num, block_num, 0] = 1.
        return self.object_features, edge_features

    def transition(self, state, action):
        object_features, edge_features = state
        delta_edge_features = model_forward(self.model, [object_features, edge_features, action])
        delta_edge_features = delta_edge_features.squeeze(0) # since only returns a batch of 1
        new_edge_state = edge_features + delta_edge_features
        new_edge_state = np.round(new_edge_state)+0
        return object_features, new_edge_state

    def is_goal_state(self, state, goal):
        # goal is logical and state is vec
        object_features, goal_edge_features = logical_to_vec_state(goal, self.num_objects)
        object_features, state_edge_features = state
        goal_idxs = np.array(np.where(goal_edge_features == 1))
        goal_reached = True
        for goal_idx in goal_idxs.T:
            # NOTE: This only works when vactorized states are 2D
            goal_reached = goal_reached and (np.round(state_edge_features[goal_idx[0]][goal_idx[1]]) == 1)
        return goal_reached

class ABCBlocksWorldGTOpt(ABCBlocksWorldGT):
    def __init__(self, num_blocks):
        super().__init__(num_blocks)

    def transition(self, state, action):
        return super().transition(state, action, optimistic=True)

    def steps_to_goal(self, state, goal_state):
        return super().steps_to_goal(state, goal_state, optimistic=True)

# In the ground truth world the state is separate from world so that we can get different
# state representations (logical and vectorized)
class LogicalState:
    def __init__(self, blocks, num_objects, table):
        self.table = table
        self.blocks = blocks
        self.stacked_blocks = []
        self.num_objects = num_objects

    def as_logical(self, debug=False):
        logical_state = []

        # stacked blocks
        if len(self.stacked_blocks) > 0:
            logical_state.append(On(self.table.num, self.blocks[self.stacked_blocks[0]].num))
            if debug:
                print(self.table.num, self.blocks[self.stacked_blocks[0]].num)
        for bottom_block_num, top_block_num in zip(self.stacked_blocks[:-1], self.stacked_blocks[1:]):
            logical_state.append(On(self.blocks[bottom_block_num].num, self.blocks[top_block_num].num))
            if debug:
                print(bottom_block_num, top_block_num)

        # remaining blocks on table
        for block_num, block in self.blocks.items():
            if block_num not in self.stacked_blocks:
                logical_state.append(On(self.table.num, block.num))
                if debug:
                    print(self.table.num, block_num)
        if debug:
            print('---')
        return logical_state

    def as_vec(self):
        logical_state = self.as_logical()
        return logical_to_vec_state(logical_state, self.num_objects)

    def copy(self):
        copy_state = LogicalState(self.blocks, self.num_objects, self.table)
        copy_state.stacked_blocks = deepcopy(self.stacked_blocks)
        return copy_state

    def is_equal(self, state):
        this_edge_features = self.as_vec()[1]
        other_edge_features = state.as_vec()[1]
        return np.array_equal(this_edge_features, other_edge_features)

### Helper Functions
def print_state(state, num_objects):
    def print_vec_state(vec_state):
        print(vec_state[1].squeeze())
    if isinstance(state, LogicalState):
        print_vec_state(state.as_vec())
    elif isinstance(state, tuple) and len(state) == 2:
        print_vec_state(state)
    elif isinstance(state, list) and isinstance(state[0], On):
        print_vec_state(logical_to_vec_state(state, num_objects))

def logical_to_vec_state(state, num_objects):
    object_features = np.expand_dims(np.arange(num_objects), 1)
    edge_features = np.zeros((num_objects, num_objects, 1))

    # edge_feature[i, j, 0] == 1 if j on i, else 0
    for predicate in state:
        bottom_i = predicate.bottom_num
        top_i = predicate.top_num
        edge_features[bottom_i, top_i, 0] = 1.

    return object_features, edge_features

def generate_random_goal(world):
    top_block_num = np.random.randint(world.min_block_num+1, world.max_block_num+1)
    return [On(top_block_num-1, top_block_num)]

'''
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
'''
