import argparse
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from learning.domains.abc_blocks.world import ABCBlocksWorldGT, ABCBlocksWorldLearned, \
                                            ABCBlocksWorldLearnedClass, print_state, \
                                            ABCBlocksWorldGTOpt, LogicalState
from learning.domains.abc_blocks.abc_blocks_data import model_forward
from learning.active.utils import GoalConditionedExperimentLogger
from planning.tree import Tree, Node

# this is a temporary HACK
from learning.evaluate.utils import vec_to_logical_state
rank_accuracy = 0

def setup_world(args):
    print('Planning with %s model.' % args.model_type)
    if args.model_type == 'learned':
        model_logger = GoalConditionedExperimentLogger(args.model_exp_path)
        model = model_logger.load_trans_model()
        if model.pred_type == 'class':
            world = ABCBlocksWorldLearnedClass(args.num_blocks, model)
        else:
            world = ABCBlocksWorldLearned(args.num_blocks, model)
        print('Using model %s.' % model_logger.exp_path)
    elif args.model_type == 'true':
        world = ABCBlocksWorldGT(args.num_blocks)
    elif args.model_type == 'opt':
        world = ABCBlocksWorldGTOpt(args.num_blocks)
    return world

def state_exists(tree, state):
    for node_id, node in tree.nodes.items():
        if isinstance(state, tuple) and np.array_equal(state[1], node.state[1]):
            return node.id
        elif isinstance(state, LogicalState) and state.is_equal(node.state):
            return node.id
    return None

def mcts(tree, world, node_value_fn, node_select_fn, node_update, args):
    for t in range(args.timeout):
        sys.stdout.write("Search progress: %i   \r" % (t) )
        sys.stdout.flush()
        parent_node = tree.traverse(node_select_fn)
        state = parent_node.state
        ## HACK
        if not isinstance(state, LogicalState):
            lstate = vec_to_logical_state(state[1], world)
        else:
            lstate = state
        ##
        new_actions = [world.random_policy(lstate) for _ in range(args.num_branches)]
        new_states = [world.transition(state, action) for action in new_actions]

        for new_action, new_state in zip(new_actions, new_states):
            node_id = state_exists(tree, new_state)
            if node_id is None: # state not in tree yet
                new_node = Node(new_state, new_action, parent_node.id)
                node_id = tree.expand(new_node)
            node_value = node_value_fn(node_id)
            node_update(node_id, node_value)
    return tree

def run(goal, args):
    world = setup_world(args)
    tree = Tree(world, goal, args)
    if args.value_fn == 'rollout':
        print('Using random model rollouts to estimate node value.')
        node_value_fn = tree.rollout
        node_select_fn = tree.get_uct_node
        node_update = tree.backpropagate
    elif args.value_fn == 'learned':
        model_logger = GoalConditionedExperimentLogger(args.model_exp_path)
        print('Using heuristic model %s.' % model_logger.exp_path)
        heur_model = model_logger.load_heur_model()
        node_value_fn = lambda node_id: tree.get_heuristic(node_id, heur_model)
        node_select_fn = tree.get_min_steps_node
        node_update = tree.update_steps
    tree = mcts(tree, world, node_value_fn, node_select_fn, node_update, args)
    found_plan = plan_from_tree(world, goal, tree, debug=False)

    logger = GoalConditionedExperimentLogger.setup_experiment_directory(args, 'planning')
    logger.save_planning_data(tree, goal, found_plan)
    print('Saved planning tree, goal and plan to %s.' % logger.exp_path)
    return found_plan, logger.exp_path, rank_accuracy

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
                        choices=['learned', 'true', 'opt'],
                        default='true',
                        help='plan with learned or ground truth model')
    parser.add_argument('--value-fn',
                        type=str,
                        choices=['rollout', 'learned'],
                        default='rollout',
                        help='use random model rollouts to estimate node value or learned value')
    parser.add_argument('--model-exp-path',
                        type=str,
                        help='Path to torch model to use during planning for learned transitions and/or learned heuristic')
    parser.add_argument('--exp-name',
                        type=str,
                        required=True,
                        help='experiment name for saving planning results')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    # for testing
    from tamp.predicates import On
    goal = [On(1, 2), On(2, 3)]

    tree = run(goal, args)
