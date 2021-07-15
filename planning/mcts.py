from collections import namedtuple
import argparse
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from planning.tree import Tree, Node
from learning.domains.abc_blocks.world import ABCBlocksWorldGT, print_state

def run(world, goal, args):
    tree = Tree(world, goal, args)
    for t in range(args.timeout):
        sys.stdout.write("Search progress: %i   \r" % (t) )
        sys.stdout.flush()
        parent_node = tree.traverse()
        state = parent_node.state
        new_actions = [world.random_policy(state) for _ in range(args.num_branches)]
        new_states = [world.transition(state, action) for action in new_actions]
        new_nodes = [Node(new_state, parent_node.id) for new_state in new_states]

        for new_action, new_node in zip(new_actions, new_nodes):
            new_node_id = tree.expand(new_node)
            rollout_value = tree.rollout(new_node_id)
            tree.backpropagate(new_node_id, rollout_value)

    return tree

def plan_from_tree(world, goal, tree):
    goal_nodes = list(filter(lambda node: world.is_goal_state(node.state, goal), tree.nodes.values()))
    print('Max value in tree is: %f.' % max([node.value for node in tree.nodes.values()]))
    if goal_nodes != []:
        print('Goal found!')
        goal_node_values = [goal_node.value for goal_node in goal_nodes]
        best_goal_node_idx = goal_node_values.index(max(goal_node_values))
        best_goal_node = goal_nodes[best_goal_node_idx]

        plan = [best_goal_node]
        node = best_goal_node
        while node.id != 0:
            node = tree.nodes[node.parent_id]
            plan = [node] + plan

        # TODO: add actions to tree
        for node in plan:
            print_state(node.state)
            print('---')
    else:
        print('Goal not found!')
