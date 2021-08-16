from collections import namedtuple
import numpy as np

from learning.domains.abc_blocks.abc_blocks_data import model_forward
from learning.domains.abc_blocks.world import logical_to_vec_state, LogicalState
# this is a temporary HACK
from learning.evaluate.utils import vec_to_logical_state
class Node:
    def __init__(self, state, action, parent_id):
        self.state = state
        self.action = action
        self.parent_id = parent_id
        self.children = []
        self.leaf = True
        self.value = 0
        self.visit_count = 0
        self.steps_to_goal = float('inf')
        #self.g = 0
        #self.h = 0

        self.id = None # set when added to tree

class Tree:
    def __init__(self, world, goal, args):
        self.world = world
        self.goal = goal

        init_state = self.world.get_init_state()
        init_node = Node(init_state, None, None)
        init_node.id = 0
        self.nodes = {0: init_node}
        self.tree_count = 1

        self.c = args.c
        self.max_ro = args.max_ro # max rollout steps

    def traverse(self, node_select_fn):
        node = self.nodes[0]
        while not node.leaf:
            node = node_select_fn(node)
        if self.world.is_goal_state(node.state, self.goal):
            return self.nodes[node.parent_id]
        return node

    # random rollout
    def rollout(self, node_id):
        state = self.nodes[node_id].state
        i = 0
        while not self.world.is_goal_state(state, self.goal) and i < self.max_ro:
            ## HACK
            if not isinstance(state, LogicalState):
                lstate = vec_to_logical_state(state[1], self.world)
            else:
                lstate = state
            ##
            if state == None:
                import pdb; pdb.set_trace()
            action = self.world.random_policy(lstate)
            state = self.world.transition(state, action)
            i += 1
        return self.world.reward(state, self.goal)

    # adjust value and increase count
    def backpropagate(self, node_id, value):
        while node_id is not None: # while not at root node
            old_count = self.nodes[node_id].visit_count
            old_value = self.nodes[node_id].value
            new_value = (old_count*old_value + value)/(old_count+1)
            self.nodes[node_id].visit_count += 1
            self.nodes[node_id].value = new_value
            node_id = self.nodes[node_id].parent_id

    def update_steps(self, node_id, value):
        self.nodes[node_id] = value
        while node_id is not None:
            self.nodes[node_id].visit_count += 1
            node_id = self.nodes[node_id].parent_id

    def get_uct_node(self, node):
        children = [self.nodes[child_id] for child_id in node.children]
        uct_values = np.zeros(len(children))
        for i, child in enumerate(children):
            node_value = child.value
            expl_bonus =  self.c*np.sqrt(np.log(child.visit_count)/child.visit_count)
            uct_values[i] = node_value + expl_bonus
        uct_node_idx = np.argmax(uct_values)
        return children[uct_node_idx]

    def get_min_steps_node(self, node):
        children = [self.nodes[child_id] for child_id in node.children]
        step_values = np.zeros(len(children))
        for i, child in enumerate(children):
            step_values[i] = child.steps_to_goal
        min_steps_node_idx = np.argmin(step_values)
        return children[min_steps_node_idx]
    '''
    def get_min_cost_node(self, node_ids):
        node_costs = [self.nodes[node_id].g+self.nodes[node_id].h for node_id in node_ids]
        min_cost_node_index = np.argmin(node_costs)
        return node_ids[min_cost_node_index]
    '''
    def expand(self, new_node):
        new_node_id = self.tree_count
        new_node.id = new_node_id
        self.nodes[new_node_id] = new_node
        self.nodes[new_node.parent_id].children += [new_node_id]
        self.nodes[new_node.parent_id].leaf = False
        self.tree_count += 1
        return new_node_id

    def get_heuristic(self, node_id, heuristic_model):
        node_state = self.nodes[node_id].state
        if isinstance(node_state, LogicalState):
            vec_state = node_state.as_vec()
        else:
            vec_state = node_state
        vec_goal_state = logical_to_vec_state(self.goal, self.world.num_objects)
        model_input = [vec_state[0], vec_state[1], vec_goal_state[1]]
        heuristic_values = model_forward(heuristic_model, model_input)
        return heuristic_values[0] # since returns batch of 1
