from collections import namedtuple
import numpy as np

class Node:
    def __init__(self, state, action, parent_id):
        self.state = state
        self.action = action
        self.parent_id = parent_id
        self.children = []
        self.leaf = True
        self.value = 0
        self.visit_count = 0

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

    def traverse(self):
        node = self.nodes[0]
        while not node.leaf:
            node = self.get_uct_node(node)
        if self.world.is_goal_state(node.state, self.goal):
            return self.nodes[node.parent_id]
        return node

    # random rollout
    def rollout(self, node_id):
        state = self.nodes[node_id].state
        i = 0
        while not self.world.is_goal_state(state, self.goal) and i < self.max_ro:
            action = self.world.random_policy(state)
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

    def get_uct_node(self, node):
        children = [self.nodes[child_id] for child_id in node.children]
        uct_values = np.zeros(len(children))
        for i, child in enumerate(children):
            node_value = child.value
            expl_bonus =  self.c*np.sqrt(np.log(child.visit_count)/child.visit_count)
            uct_values[i] = node_value + expl_bonus
        uct_node_idx = np.argmax(uct_values)
        return children[uct_node_idx]

    def expand(self, new_node):
        new_node_id = self.tree_count
        new_node.id = new_node_id
        self.nodes[new_node_id] = new_node
        self.nodes[new_node.parent_id].children += [new_node_id]
        self.nodes[new_node.parent_id].leaf = False
        self.tree_count += 1
        return new_node_id
