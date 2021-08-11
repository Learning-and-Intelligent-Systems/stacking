import argparse
import numpy as np
import matplotlib.pyplot as plt
import pydot

from learning.domains.abc_blocks.abc_blocks_data import model_forward
from learning.domains.abc_blocks.world import ABCBlocksWorldGT, LogicalState
from learning.active.utils import GoalConditionedExperimentLogger

# this generally doesn't work, but works when the pred_type == class
# because the edge states are all valid states
def vec_to_logical_state(edge_features, world):
    state = LogicalState(world._blocks, world.num_blocks, world.table)
    state.stacked_blocks = vec_state_to_stacked_blocks(edge_features)
    return state

def vec_state_to_stacked_blocks(edge_features):
    stacked_blocks = set()
    num_objects = edge_features.shape[0]
    for bottom_i in range(num_objects):
        for top_i in range(num_objects):
            if edge_features[bottom_i, top_i] == 1.:
                if bottom_i != 0 and top_i != 0: # 0 is the table
                    stacked_blocks.add(bottom_i)
                    stacked_blocks.add(top_i)
    stacked_blocks = list(stacked_blocks)
    stacked_blocks.sort()
    return stacked_blocks

def stacked_blocks_to_str(stacked_blocks):
    if len(stacked_blocks) > 0:
        str_state = ''
        str_state += str(stacked_blocks[0])
        for block_id in stacked_blocks[1:]:
            str_state += '/'+str(block_id)
    else:
        str_state = '-'
    return str_state

# learning the delta state saves states as vectors
# all others (including learning a classifier) save the state as a LogicalState
def node_to_label(node):
    if isinstance(node.state, tuple): # vec state
        str_state = str(np.round(node.state[1].squeeze())) + 0 # get rid of -0s
    else: # logical state
        stacked_blocks_to_str(node.state.stacked_blocks)
    str_state += '\n'+str(node.value)
    return str_state

def generate_dot_graph(plan_path, model_path):
    plan_logger = GoalConditionedExperimentLogger(plan_path)
    model_logger = GoalConditionedExperimentLogger(model_path)
    goal = plan_logger.load_plan_goal()
    tree = plan_logger.load_plan_tree()
    plan = plan_logger.load_final_plan()

    plan_ids = []
    if plan is not None:
        plan_ids = [node.id for node in plan]
    graph = pydot.Dot("my_graph", graph_type="graph")
    # add goal node TODO: make more general
    graph.add_node(pydot.Node(-1, label=str(goal[0].bottom_num)+'/'+str(goal[0].top_num), color='red'))
    # make all nodes
    for node_id, node in tree.nodes.items():
        label = node_to_label(node)
        if node_id in plan_ids:
            graph.add_node(pydot.Node(node_id, label=label, color="green"))
        else:
            graph.add_node(pydot.Node(node_id, label=label))
    # add all edges
    for node_id, node in tree.nodes.items():
        for child_id in node.children:
            graph.add_edge(pydot.Edge(node_id, child_id))
    plan_logger.save_dot_graph(dot_graph)

from operator import itemgetter
def join_strs(str_list, indices):
    if len(indices) == 1:
        return str_list[indices[0]]
    else:
        return ' , '.join(itemgetter(*indices)(str_list))

def plot_horiz_bars(transitions, plot_inds, bar_text_inds, color=False, plot_title=None, y_label=None):

    plot_data = {}
    plot_text = {}
    for transition in transitions:
        key = join_strs(transition, plot_inds)
        bar_text = join_strs(transition, bar_text_inds)
        if key in plot_data:
            plot_data[key] += 1
            plot_text[key].add(bar_text)
        else:
            plot_data[key] = 1
            plot_text[key] = set()
            plot_text[key].add(bar_text)

    bar_color = lambda vals: 'green' if list(vals)[0]=='1' else 'red'
    bar_colors = [bar_color(v) for v in plot_text.values()]

    fig, ax = plt.subplots()
    if color:
        bar_plot = ax.barh(np.arange(len(plot_data)), plot_data.values(), align='center', color=bar_colors)
    else:
        bar_plot = ax.barh(np.arange(len(plot_data)), plot_data.values(), align='center')
    ax.set_yticks(np.arange(len(plot_data)))
    ax.set_yticklabels(plot_data.keys())
    if y_label:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel('Transitions')
    ax.set_xlabel('Frequency')
    if plot_title:
        ax.set_title(plot_title)

    # show text on bars
    for rect, text_set in zip(bar_plot, plot_text.values()):
        ax.text(0,
                rect.get_y() + rect.get_height()/2,
                ' // '.join(list(text_set)),
                ha='left',
                va='center',
                rotation=0)

def evaluate_dataset(model_path):
    model_logger = GoalConditionedExperimentLogger(model_path)
    model_args = model_logger.load_args()

    dataset_path = model_args.dataset_exp_path
    dataset_logger = GoalConditionedExperimentLogger(dataset_path)
    dataset = dataset_logger.load_trans_dataset()
    dataset.set_pred_type(model_args.pred_type)

    dataset_args = dataset_logger.load_args()
    world = ABCBlocksWorldGT(dataset_args.num_blocks)

    model = model_logger.load_trans_model()

    if dataset.pred_type == 'class': # for now not visualizing delta state transitions
        # plot training dataset
        transitions = []
        for (object_features, edge_features, action), gt_pred in dataset:
            model_pred_float = model_forward(model, [object_features, edge_features, action])
            model_pred = model_pred_float.round()
            correct = str(int(model_pred == gt_pred.numpy()))
            action = [int(a) for a in action]
            state = vec_to_logical_state(edge_features, world)
            next_state = world.transition(state, action)
            next_opt_state = world.transition(state, action, optimistic=True)

            str_state = stacked_blocks_to_str(state.stacked_blocks)
            str_action = '%i/%i' % (action[0], action[1])
            str_next_state = stacked_blocks_to_str(next_state.stacked_blocks)
            str_pred = '%i' % gt_pred
            str_opt_next_state = stacked_blocks_to_str(next_opt_state.stacked_blocks)
            transitions.append([str_state, str_action, str_next_state, str_pred, str_opt_next_state, correct])

        all_trans_keys = [0,1,2,3]
        all_trans_bar_text = [4]
        plot_horiz_bars(transitions, all_trans_keys, all_trans_bar_text)

        classification_keys = [3]
        classification_bar_text = [0,1,2]
        plot_horiz_bars(transitions, classification_keys, classification_bar_text)

        init_state_keys = [0,3]
        init_state_bar_text = [1,2]
        plot_horiz_bars(transitions, init_state_keys, init_state_bar_text)

        acc_keys = [0,1]
        acc_bar_text = [5]
        plot_horiz_bars(transitions, acc_keys, acc_bar_text)

def evaluate_model(model_path, test_dataset_path, plot_title):
    model_logger = GoalConditionedExperimentLogger(model_path)
    model_args = model_logger.load_args()
    model = model_logger.load_trans_model()

    test_dataset_logger = GoalConditionedExperimentLogger(test_dataset_path)
    test_dataset_args = test_dataset_logger.load_args()
    test_dataset = test_dataset_logger.load_trans_dataset()
    test_dataset.set_pred_type(model_args.pred_type)
    test_world = ABCBlocksWorldGT(test_dataset_args.num_blocks)

    if model_args.pred_type == 'class': # for now not visualizing delta state transitions
        # plot accuracy
        transitions = []
        for (object_features, edge_features, action), gt_pred in test_dataset:
            model_pred_float = model_forward(model, [object_features, edge_features, action])
            model_pred = model_pred_float.round()
            action = [int(a) for a in action]
            state = vec_to_logical_state(edge_features, test_world)
            next_state = test_world.transition(state, action)
            next_opt_state = test_world.transition(state, action, optimistic=True)

            str_state = stacked_blocks_to_str(state.stacked_blocks)
            str_action = '%i/%i' % (action[0], action[1])
            str_next_state = stacked_blocks_to_str(next_state.stacked_blocks)
            str_gt_pred = '%i' % gt_pred
            str_pred = '%i' % model_pred
            str_opt_next_state = stacked_blocks_to_str(next_opt_state.stacked_blocks)
            str_acc = str(int(model_pred == gt_pred.numpy()))
            transitions_names = ['state', 'action', 'next state', 'gt pred', 'model pred', 'optimistic next state', 'model accuracy']
            transitions.append([str_state,
                                str_action,
                                str_next_state,
                                str_gt_pred,
                                str_pred,
                                str_opt_next_state,
                                str_acc])

        acc_keys = [0,1,2,3,4,5]
        acc_bar_text = [6]
        y_label = join_strs(transitions_names, acc_keys)
        plot_horiz_bars(transitions, acc_keys, acc_bar_text, color=True, plot_title=plot_title, y_label=y_label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path',
                        type=str,
                        help='what planning run to be analyzed in /exp_goals, else use given hard coded list below')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    parser.add_argument('--gen-dot-graphs',
                        action='store_true')
    parser.add_argument('--gen-dataset-stats',
                        action='store_true')
    parser.add_argument('--gen-model-accuracy',
                        action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    plan_paths = []
    model_paths = ['learning/experiments/logs/models/large-train-set-20210810-160955',
                    'learning/experiments/logs/models/large-train-set-20210810-161047',
                    'learning/experiments/logs/models/large-train-set-20210810-161118',
                    'learning/experiments/logs/models/large-train-set-20210810-161223']

    # Expert Test Sets
    #test_dataset_path = 'learning/experiments/logs/datasets/large-test-set-2-20210810-161843'
    #test_dataset_path = 'learning/experiments/logs/datasets/large-test-set-3-20210810-161852'
    #test_dataset_path = 'learning/experiments/logs/datasets/large-test-set-4-20210810-161859'
    #test_dataset_path = 'learning/experiments/logs/datasets/large-test-set-5-20210810-161906'
    #test_dataset_path = 'learning/experiments/logs/datasets/large-test-set-6-20210810-161913'

    # Random Test Sets (all 4 block sets used to train 4 model)
    #test_dataset_path = 'learning/experiments/logs/datasets/large-train-set-20210810-155116'
    #test_dataset_path = 'learning/experiments/logs/datasets/large-train-set-20210810-160411'
    #test_dataset_path = 'learning/experiments/logs/datasets/large-train-set-20210810-160512'
    #test_dataset_path = 'learning/experiments/logs/datasets/large-train-set-20210810-160519'

    # Random Test Sets (2,3,4,5,6 num blocks)
    #test_dataset_path = 'learning/experiments/logs/datasets/large-test-set-random-2-20210810-165325'
    test_dataset_path = 'learning/experiments/logs/datasets/large-test-set-random-3-20210810-165531'
    #test_dataset_path = 'learning/experiments/logs/datasets/large-test-set-random-4-20210810-165541'
    #test_dataset_path = 'learning/experiments/logs/datasets/large-test-set-random-5-20210810-165608'
    #test_dataset_path = 'learning/experiments/logs/datasets/large-test-set-random-6-20210810-165910'

    if args.exp_path:
        logger = GoalConditionedExperimentLogger(args.exp_path)
        plan_plot_data = logger.load_plot_data()
        plot_data_paths = plan_plot_data[0]
        plot_data_successes = plan_plot_data[1]
        for method, method_data in plot_data_paths.items():
            for plan_num_blocks, num_blocks_data in method_data.items():
                for model_path, plan_paths in num_blocks_data.items():
                    for plan_path in plan_paths:
                        if args.gen_dot_graphs:
                            print('Generating dot graphs for plans in %s' % args.exp_path)
                            generate_dot_graph(plan_path, model_path)
                        if args.gen_dataset_stats:
                            evaluate_dataset(model_path)
                            print('Plotting model accuracy for models in %s. Currently only works for classification.' % args.exp_path)
                        if args.gen_model_accuracy:
                            print('Can only plot model accuracy with hard coded paths')
    else:
        for plan_path in plan_paths:
            plan_logger = GoalConditionedExperimentLogger(plan_path)
            plan_args = plan_logger.load_args()
            model_path = plan_args.model_exp_path
            if args.gen_dot_graphs:
                print('Generating dot graphs for hardcoded plan_paths in file.')
                generate_dot_graph(plan_path, model_path)
        for mi, model_path in enumerate(model_paths):
            if args.gen_dataset_stats:
                print('Plotting model accuracy for hardcoded model_paths in file. Currently only works for classification.')
                evaluate_dataset(model_path)
            if args.gen_model_accuracy:
                evaluate_model(model_path, test_dataset_path, 'Model %i' % mi)

    plt.show()
