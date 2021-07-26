import argparse
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from learning.domains.abc_blocks.world import ABCBlocksWorldGT, ABCBlocksWorldLearned, print_state
from learning.active.utils import GoalConditionedExperimentLogger
from planning.tree import Tree, Node

def setup_world(args):
    print('Planning with %s model.' % args.model_type)
    if args.model_type == 'learned':
        model_logger = GoalConditionedExperimentLogger(args.model_exp_path)
        model = model_logger.load_trans_model()
        world = ABCBlocksWorldLearned(args.num_blocks, model)
        print('Using model %s.' % model_logger.exp_path)
    elif args.model_type == 'true':
        world = ABCBlocksWorldGT(args.num_blocks)
        print('Planning with %i blocks.' % args.num_blocks)
    return world

def search(tree, world, node_value_fn, node_select_fn, args):
    for t in range(args.timeout):
        sys.stdout.write("Search progress: %i   \r" % (t) )
        sys.stdout.flush()
        parent_node = tree.traverse(node_select_fn)
        state = parent_node.state
        new_actions = [world.random_policy(state) for _ in range(args.num_branches)]
        new_states = [world.transition(state, action) for action in new_actions]
        new_nodes = [Node(new_state, new_action, parent_node.id) for (new_state, new_action) \
                                in zip(new_states, new_actions)]
        for new_node in new_nodes:
            new_node_id = tree.expand(new_node)
            rollout_value = node_value_fn(new_node_id)
            tree.backpropagate(new_node_id, rollout_value)
    return tree

def run(goal, args):
    world = setup_world(args)
    tree = Tree(world, goal, args)
    print('Planning method is %s search.' % args.search_mode)
    if args.search_mode == 'mcts':
        node_value_fn = tree.rollout
        node_select_fn = tree.get_uct_node
    elif args.search_mode == 'heuristic':
        model_logger = GoalConditionedExperimentLogger(args.model_exp_path)
        print('Using heuristic model %s.' % model_logger.exp_path)
        heur_model = model_logger.load_heur_model()
        node_value_fn = lambda node_id: tree.get_heuristic(node_id, heur_model)
        node_select_fn = tree.get_min_value_node
    tree = search(tree, world, node_value_fn, node_select_fn, args)
    found_plan = plan_from_tree(world, goal, tree, debug=False)

    logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'planning')
    logger.save_planning_data(tree, goal, found_plan)
    print('Saved planning tree, goal and plan to %s.' % logger.exp_path)
    return found_plan, logger.exp_path

def plan_from_tree(world, goal, tree, debug=False):
    found_plan = None
    goal_nodes = list(filter(lambda node: world.is_goal_state(node.state, goal), tree.nodes.values()))
    print('Max value in tree is: %f.' % max([node.value for node in tree.nodes.values()]))
    if goal_nodes != []:
        print('Goal found!')
        goal_node_values = [goal_node.value for goal_node in goal_nodes]
        best_goal_node_idx = goal_node_values.index(max(goal_node_values))
        best_goal_node = goal_nodes[best_goal_node_idx]

        found_plan = [best_goal_node]
        node = best_goal_node
        while node.id != 0:
            node = tree.nodes[node.parent_id]
            found_plan = [node] + found_plan

        if debug:
            for node in found_plan:
                print(node.action)
                print('---')
                print_state(node.state, world.num_objects)
                print('---')

    else:
        print('Goal not found!')
    return found_plan

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-blocks',
                        type=int,
                        default=3)
    parser.add_argument('--num-branches',
                        type=int,
                        default=10,
                        help='number of actions to try from each node')
    parser.add_argument('--timeout',
                        type=int,
                        default=100,
                        help='Iterations to run MCTS')
    parser.add_argument('--c',
                        type=int,
                        default=.01,
                        help='UCB parameter to balance exploration and exploitation')
    parser.add_argument('--max-ro',
                        type=int,
                        default=10,
                        help='Maximum number of random rollout steps')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--model-type',
                        type=str,
                        choices=['learned', 'true'],
                        default='true',
                        help='plan with learned or ground truth model')
    parser.add_argument('--model-exp-path',
                        type=str,
                        help='Path to torch model to use during planning for learned transitions and/or learned heuristic')
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='experiment name for saving planning results')
    parser.add_argument('--search-mode',
                        type=str,
                        required=True,
                        choices = ['mcts', 'heuristic'],
                        help='type of search')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    # for testing
    from tamp.predicates import On
    goal = [On(1, 2), On(2, 3)]

    tree = run(goal, args)
